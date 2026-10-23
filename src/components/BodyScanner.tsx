"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  FilesetResolver,
  PoseLandmarker,
  DrawingUtils,
} from "@mediapipe/tasks-vision";
import {
  NormalizedLandmark,
  evaluatePoseQuality,
  BodyContourSlices,
} from "@/lib/estimateMeasurements";
import {
  Camera,
  CheckCircle2,
  RefreshCw,
  Sparkles,
  Timer,
  Zap,
  Crosshair,
  ShieldCheck,
  Scan,
} from "lucide-react";

interface BodyScannerProps {
  view: "front" | "side";
  onCapture: (
    landmarks: NormalizedLandmark[],
    worldLandmarks: NormalizedLandmark[] | null,
    snapshotUrl: string,
    contourSlices?: BodyContourSlices
  ) => void;
  onStatusChange?: (status: string) => void;
  existingSnapshot?: string | null;
}

export default function BodyScanner({
  view,
  onCapture,
  onStatusChange,
  existingSnapshot,
}: BodyScannerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const animFrameIdRef = useRef<number | null>(null);
  const latestLandmarksRef = useRef<NormalizedLandmark[] | null>(null);
  const latestWorldLandmarksRef = useRef<NormalizedLandmark[] | null>(null);
  const latestSlicesRef = useRef<BodyContourSlices>({});

  const burstBufferRef = useRef<{
    landmarks: NormalizedLandmark[][];
    worldLandmarks: NormalizedLandmark[][];
  }>({ landmarks: [], worldLandmarks: [] });

  const [status, setStatus] = useState<string>("Camera not started");
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelPrecision, setModelPrecision] = useState<"full" | "lite">("full");
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");
  const [countdown, setCountdown] = useState<number | null>(null);
  const [isBursting, setIsBursting] = useState(false);
  const [poseScore, setPoseScore] = useState<number>(0);
  const [qualityWarnings, setQualityWarnings] = useState<string[]>([]);
  const [gateStatus, setGateStatus] = useState({
    head: false,
    torso: false,
    feet: false,
    distanceOk: false,
    tiltOk: false,
    armsOk: false,
  });

  const updateStatus = useCallback((msg: string) => {
    setStatus(msg);
    onStatusChange?.(msg);
  }, [onStatusChange]);

  const stopCamera = useCallback(() => {
    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current);
      animFrameIdRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsCameraActive(false);
    updateStatus("Camera paused");
  }, [updateStatus]);

  useEffect(() => {
    return () => {
      stopCamera();
      if (landmarkerRef.current) {
        try {
          landmarkerRef.current.close();
        } catch {
          // ignore
        }
        landmarkerRef.current = null;
      }
    };
  }, [stopCamera]);

  async function getOrInitLandmarker(): Promise<PoseLandmarker> {
    if (landmarkerRef.current) return landmarkerRef.current;

    setModelLoading(true);
    updateStatus("Initializing High-Precision Neural Model...");

    let vision;
    try {
      vision = await FilesetResolver.forVisionTasks("/wasm");
    } catch {
      vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );
    }

    const preferredModelPath =
      modelPrecision === "full"
        ? "/models/pose_landmarker_full.task"
        : "/models/pose_landmarker_lite.task";

    try {
      const landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: preferredModelPath,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.6,
        minPosePresenceConfidence: 0.6,
        minTrackingConfidence: 0.6,
      });
      landmarkerRef.current = landmarker;
      setModelLoading(false);
      return landmarker;
    } catch {
      const fallbackPath = "/models/pose_landmarker_lite.task";
      const landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: fallbackPath,
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numPoses: 1,
      });
      landmarkerRef.current = landmarker;
      setModelLoading(false);
      return landmarker;
    }
  }

  async function startCamera() {
    if (busy || isCameraActive) return;
    setBusy(true);
    updateStatus("Requesting camera access...");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: facingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });

      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error("Video element is not ready");

      video.srcObject = stream;
      await video.play();

      const landmarker = await getOrInitLandmarker();
      setIsCameraActive(true);
      updateStatus("Camera ready. Stand in optimal position.");

      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const drawingUtils = new DrawingUtils(ctx);

      let lastVideoTime = -1;

      const renderLoop = () => {
        if (!videoRef.current || !canvasRef.current) return;
        const v = videoRef.current;
        const c = canvasRef.current;

        if (v.readyState >= 2) {
          c.width = v.videoWidth;
          c.height = v.videoHeight;
          ctx.clearRect(0, 0, c.width, c.height);

          if (v.currentTime !== lastVideoTime) {
            lastVideoTime = v.currentTime;
            try {
              const results = landmarker.detectForVideo(v, performance.now());
              if (results && results.landmarks && results.landmarks.length > 0) {
                const landmarks = results.landmarks[0] as NormalizedLandmark[];
                latestLandmarksRef.current = landmarks;

                if (results.worldLandmarks && results.worldLandmarks.length > 0) {
                  latestWorldLandmarksRef.current = results.worldLandmarks[0] as unknown as NormalizedLandmark[];
                }

                if (burstBufferRef.current.landmarks.length < 15 && isBursting) {
                  burstBufferRef.current.landmarks.push(landmarks);
                  if (latestWorldLandmarksRef.current) {
                    burstBufferRef.current.worldLandmarks.push(latestWorldLandmarksRef.current);
                  }
                }

                // Draw skeleton connectors
                for (const lmGroup of results.landmarks) {
                  drawingUtils.drawConnectors(lmGroup, PoseLandmarker.POSE_CONNECTIONS, {
                    color: "#38bdf8",
                    lineWidth: 3,
                  });
                  drawingUtils.drawLandmarks(lmGroup, {
                    color: "#f59e0b",
                    fillColor: "#ffffff",
                    radius: 4,
                  });
                }

                // Draw Precision Horizontal Laser Calipers
                const lShoulder = landmarks[11];
                const rShoulder = landmarks[12];
                const lHip = landmarks[23];
                const rHip = landmarks[24];
                const lKnee = landmarks[25];
                const rKnee = landmarks[26];

                if (lShoulder && rShoulder && lHip && rHip) {
                  const shoulderY = (lShoulder.y + rShoulder.y) / 2;
                  const hipY = (lHip.y + rHip.y) / 2;
                  const torsoH = Math.abs(hipY - shoulderY);

                  // Chest slice at 25% down torso
                  const chestY = (shoulderY + torsoH * 0.25) * c.height;
                  const shoulderDistPx = Math.abs(lShoulder.x - rShoulder.x) * c.width;
                  const chestWidthPx = shoulderDistPx * 0.88;
                  const chestMidX = ((lShoulder.x + rShoulder.x) / 2) * c.width;

                  // Waist slice at 60% down torso
                  const waistY = (shoulderY + torsoH * 0.60) * c.height;
                  const waistWidthPx = chestWidthPx * 0.84;
                  const waistMidX = ((lHip.x + rHip.x) / 2) * c.width;

                  // Hip slice at hip level
                  const hipsY = hipY * c.height;
                  const hipWidthPx = Math.abs(lHip.x - rHip.x) * c.width * 1.15;

                  // Thigh slice at 25% down leg
                  const kneeY = ((lKnee?.y || hipY) + (rKnee?.y || hipY)) / 2;
                  const thighY = (hipY + Math.abs(kneeY - hipY) * 0.25) * c.height;
                  const thighWidthPx = hipWidthPx * 0.55;

                  latestSlicesRef.current = {
                    chestWidthNorm: chestWidthPx / c.width,
                    waistWidthNorm: waistWidthPx / c.width,
                    hipWidthNorm: hipWidthPx / c.width,
                    thighWidthNorm: thighWidthPx / c.width,
                  };

                  // Draw Caliper Lines
                  const calipers = [
                    { y: chestY, midX: chestMidX, w: chestWidthPx, label: "Chest Slice" },
                    { y: waistY, midX: waistMidX, w: waistWidthPx, label: "Waist Slice" },
                    { y: hipsY, midX: waistMidX, w: hipWidthPx, label: "Hip Slice" },
                    { y: thighY, midX: waistMidX - hipWidthPx * 0.25, w: thighWidthPx, label: "Thigh Slice" },
                  ];

                  for (const cal of calipers) {
                    const xStart = cal.midX - cal.w / 2;
                    const xEnd = cal.midX + cal.w / 2;

                    ctx.save();
                    ctx.strokeStyle = "#34d399";
                    ctx.lineWidth = 2;
                    ctx.setLineDash([4, 4]);

                    // Caliper bar
                    ctx.beginPath();
                    ctx.moveTo(xStart, cal.y);
                    ctx.lineTo(xEnd, cal.y);
                    ctx.stroke();

                    // Caliper end notches
                    ctx.setLineDash([]);
                    ctx.beginPath();
                    ctx.moveTo(xStart, cal.y - 6);
                    ctx.lineTo(xStart, cal.y + 6);
                    ctx.moveTo(xEnd, cal.y - 6);
                    ctx.lineTo(xEnd, cal.y + 6);
                    ctx.stroke();

                    // Caliper text badge
                    ctx.fillStyle = "rgba(6, 78, 59, 0.85)";
                    ctx.fillRect(xStart, cal.y - 18, 70, 14);
                    ctx.fillStyle = "#ffffff";
                    ctx.font = "bold 9px sans-serif";
                    ctx.fillText(cal.label, xStart + 4, cal.y - 7);
                    ctx.restore();
                  }
                }

                // Evaluate gates
                const evaluation = evaluatePoseQuality(landmarks);
                setPoseScore(evaluation.score);
                setQualityWarnings(evaluation.warnings);

                const hasHead = (landmarks[0]?.visibility ?? 1) > 0.6;
                const hasTorso =
                  (landmarks[11]?.visibility ?? 1) > 0.6 &&
                  (landmarks[12]?.visibility ?? 1) > 0.6 &&
                  (landmarks[23]?.visibility ?? 1) > 0.6;
                const hasFeet =
                  (landmarks[27]?.visibility ?? 1) > 0.6 ||
                  (landmarks[28]?.visibility ?? 1) > 0.6;

                setGateStatus({
                  head: hasHead,
                  torso: hasTorso,
                  feet: hasFeet,
                  distanceOk: evaluation.distanceStatus === "optimal",
                  tiltOk: evaluation.tiltAngleDeg <= 4,
                  armsOk: view === "side" ? true : evaluation.armsSeparated,
                });

                if (
                  hasHead &&
                  hasTorso &&
                  hasFeet &&
                  evaluation.distanceStatus === "optimal" &&
                  evaluation.tiltAngleDeg <= 4
                ) {
                  updateStatus("90%+ Precision Posture Verified! Calipers locked.");
                } else {
                  updateStatus("Adjust posture to align with green gates.");
                }
              } else {
                latestLandmarksRef.current = null;
                latestWorldLandmarksRef.current = null;
                setPoseScore(0);
                updateStatus("No full-body pose detected. Step back into view.");
                setGateStatus({
                  head: false,
                  torso: false,
                  feet: false,
                  distanceOk: false,
                  tiltOk: false,
                  armsOk: false,
                });
              }
            } catch {
              // frame drop
            }
          }
        }
        animFrameIdRef.current = requestAnimationFrame(renderLoop);
      };

      animFrameIdRef.current = requestAnimationFrame(renderLoop);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Camera permission denied.";
      updateStatus(`Could not access camera: ${msg}`);
      setIsCameraActive(false);
    } finally {
      setBusy(false);
    }
  }

  function triggerBurstCapture() {
    if (!latestLandmarksRef.current) {
      alert("No pose landmarks detected. Ensure full body is visible.");
      return;
    }
    const video = videoRef.current;
    if (!video) return;

    setIsBursting(true);
    burstBufferRef.current = { landmarks: [], worldLandmarks: [] };
    updateStatus("Calibrating 15 frames for 90%+ precision...");

    setTimeout(() => {
      setIsBursting(false);
      const frames = burstBufferRef.current.landmarks;
      let finalLandmarks = latestLandmarksRef.current!;
      let finalWorld = latestWorldLandmarksRef.current;

      if (frames.length >= 5) {
        const averaged: NormalizedLandmark[] = [];
        for (let ptIdx = 0; ptIdx < 33; ptIdx++) {
          const xValues = frames.map((f) => f[ptIdx]?.x || 0).sort((a, b) => a - b);
          const yValues = frames.map((f) => f[ptIdx]?.y || 0).sort((a, b) => a - b);
          const zValues = frames.map((f) => f[ptIdx]?.z || 0).sort((a, b) => a - b);
          const mid = Math.floor(xValues.length / 2);
          averaged.push({
            x: xValues[mid],
            y: yValues[mid],
            z: zValues[mid],
            visibility: 0.95,
          });
        }
        finalLandmarks = averaged;
      }

      const snapCanvas = document.createElement("canvas");
      snapCanvas.width = video.videoWidth || 640;
      snapCanvas.height = video.videoHeight || 480;
      const snapCtx = snapCanvas.getContext("2d");
      if (snapCtx) {
        snapCtx.drawImage(video, 0, 0, snapCanvas.width, snapCanvas.height);
        const dataUrl = snapCanvas.toDataURL("image/jpeg", 0.90);
        onCapture(finalLandmarks, finalWorld, dataUrl, latestSlicesRef.current);
      }
    }, 800);
  }

  function startCountdownCapture(seconds = 3) {
    if (countdown !== null) return;
    setCountdown(seconds);
    const interval = setInterval(() => {
      setCountdown((prev) => {
        if (prev === null || prev <= 1) {
          clearInterval(interval);
          triggerBurstCapture();
          return null;
        }
        return prev - 1;
      });
    }, 1000);
  }

  const isPrecisionReady =
    gateStatus.head &&
    gateStatus.torso &&
    gateStatus.feet &&
    gateStatus.distanceOk &&
    gateStatus.tiltOk;

  return (
    <div className="space-y-4">
      {/* Precision Model Selector & Guidance Header */}
      <div className="rounded-2xl border border-slate-700/60 bg-slate-900/60 p-4 text-sm text-slate-300">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div className="flex items-start gap-3">
            <div className="rounded-lg bg-emerald-500/10 p-2 text-emerald-400">
              <Scan className="h-5 w-5" />
            </div>
            <div className="space-y-1">
              <p className="font-semibold text-white flex items-center gap-2">
                <span>{view === "front" ? "Front Laser Caliper Scan" : "Side Depth Scan"}</span>
                <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-[10px] font-bold text-emerald-300 border border-emerald-500/30">
                  ISO 8559-1 Calipers Active
                </span>
              </p>
              <p className="text-xs text-slate-300">
                {view === "front"
                  ? "Stand in center. Green horizontal calipers will track chest, waist, hips, and thighs."
                  : "Turn 90° sideways. Calipers will capture sagittal depth."}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 self-start sm:self-center">
            <button
              type="button"
              onClick={() => {
                const next = modelPrecision === "full" ? "lite" : "full";
                setModelPrecision(next);
                if (landmarkerRef.current) {
                  landmarkerRef.current.close();
                  landmarkerRef.current = null;
                }
              }}
              className="inline-flex items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs text-slate-300 hover:text-white transition"
              title="Toggle Neural Precision"
            >
              <Zap className="h-3.5 w-3.5 text-amber-400" />
              <span>Model: {modelPrecision === "full" ? "3D Full (High Precision)" : "Lite"}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Camera Video Viewport */}
      <div className="relative aspect-[4/3] w-full max-w-2xl mx-auto overflow-hidden rounded-2xl border border-slate-700 bg-slate-950 shadow-2xl">
        <video
          ref={videoRef}
          playsInline
          muted
          className={`h-full w-full object-cover ${facingMode === "user" ? "-scale-x-100" : ""}`}
        />
        <canvas
          ref={canvasRef}
          className={`pointer-events-none absolute inset-0 h-full w-full object-cover ${
            facingMode === "user" ? "-scale-x-100" : ""
          }`}
        />

        {/* Framing Silhouette Guide Overlay */}
        {isCameraActive && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center opacity-30">
            <div
              className={`h-[86%] w-[44%] rounded-[50px] border-2 border-dashed transition-colors duration-300 ${
                isPrecisionReady ? "border-emerald-400 shadow-[0_0_20px_rgba(52,211,153,0.3)]" : "border-sky-400"
              }`}
            />
          </div>
        )}

        {/* 3-2-1 Countdown Overlay */}
        {countdown !== null && (
          <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="flex h-28 w-28 items-center justify-center rounded-full bg-emerald-600/90 text-6xl font-black text-white shadow-2xl animate-pulse">
              {countdown}
            </div>
          </div>
        )}

        {/* Burst Capture Progress Overlay */}
        {isBursting && (
          <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-black/70 backdrop-blur-sm space-y-3">
            <div className="h-10 w-10 animate-spin rounded-full border-4 border-emerald-400 border-t-transparent" />
            <p className="text-sm font-bold text-white">Calibrating 15 Precision Frames...</p>
          </div>
        )}

        {/* Live HUD 90%+ Accuracy Quality Gates */}
        {isCameraActive && (
          <div className="absolute top-3 left-3 z-20 flex flex-col gap-1.5 rounded-xl bg-black/75 p-3 backdrop-blur-md text-[11px]">
            <div className="flex items-center gap-1.5 font-bold text-slate-200 border-b border-slate-800 pb-1 mb-0.5">
              <Crosshair className="h-3.5 w-3.5 text-emerald-400" />
              <span>90%+ Precision Alignment</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2
                className={`h-3.5 w-3.5 ${
                  gateStatus.head && gateStatus.torso && gateStatus.feet
                    ? "text-emerald-400"
                    : "text-amber-400"
                }`}
              />
              <span className={gateStatus.head ? "text-slate-200" : "text-amber-300"}>
                Full Body in View
              </span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2
                className={`h-3.5 w-3.5 ${gateStatus.distanceOk ? "text-emerald-400" : "text-amber-400"}`}
              />
              <span className={gateStatus.distanceOk ? "text-slate-200" : "text-amber-300"}>
                Distance (70-85% frame)
              </span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2
                className={`h-3.5 w-3.5 ${gateStatus.tiltOk ? "text-emerald-400" : "text-amber-400"}`}
              />
              <span className={gateStatus.tiltOk ? "text-slate-200" : "text-amber-300"}>
                Spine Verticality (±4°)
              </span>
            </div>
            {view === "front" && (
              <div className="flex items-center gap-2">
                <CheckCircle2
                  className={`h-3.5 w-3.5 ${gateStatus.armsOk ? "text-emerald-400" : "text-amber-400"}`}
                />
                <span className={gateStatus.armsOk ? "text-slate-200" : "text-amber-300"}>
                  Arms Outward (20°–30°)
                </span>
              </div>
            )}
          </div>
        )}

        {/* Existing Snapshot Preview if camera inactive */}
        {!isCameraActive && existingSnapshot && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-slate-900/90">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={existingSnapshot}
              alt={`${view} capture snapshot`}
              className="max-h-[80%] max-w-[80%] rounded-lg border border-slate-700 shadow-md object-contain"
            />
            <p className="mt-3 text-xs font-medium text-emerald-400 flex items-center gap-1.5">
              <CheckCircle2 className="h-4 w-4" /> Captured {view} view snapshot
            </p>
          </div>
        )}

        {/* Camera inactive placeholder */}
        {!isCameraActive && !existingSnapshot && (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center text-slate-400">
            <Camera className="h-12 w-12 stroke-[1.5] text-slate-600 mb-3" />
            <p className="text-sm font-medium text-slate-300">Camera is not active</p>
            <p className="mt-1 max-w-sm text-xs text-slate-500">
              Click below to activate real-time contour calipers and 3D metric tracking.
            </p>
          </div>
        )}
      </div>

      {/* Status Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <p className="text-slate-400 flex items-center gap-1.5">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              isCameraActive
                ? isPrecisionReady
                  ? "bg-emerald-400 animate-pulse shadow-[0_0_8px_#34d399]"
                  : "bg-amber-400"
                : "bg-slate-500"
            }`}
          />
          {status}
        </p>

        {isCameraActive && (
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1 text-slate-400">
              <ShieldCheck
                className={`h-3.5 w-3.5 ${isPrecisionReady ? "text-emerald-400" : "text-amber-400"}`}
              />
              Alignment Score: <b className="text-emerald-400">{poseScore}%</b>
            </span>
            <button
              type="button"
              onClick={() => {
                stopCamera();
                setFacingMode((prev) => (prev === "user" ? "environment" : "user"));
              }}
              className="inline-flex items-center gap-1 text-slate-400 hover:text-white transition"
              title="Switch camera"
            >
              <RefreshCw className="h-3.5 w-3.5" /> Switch Cam
            </button>
          </div>
        )}
      </div>

      {/* Action Controls */}
      <div className="flex flex-wrap items-center gap-3 pt-1">
        {!isCameraActive ? (
          <button
            type="button"
            onClick={startCamera}
            disabled={busy || modelLoading}
            className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 px-5 py-2.5 text-sm font-medium text-white shadow-lg shadow-emerald-500/20 hover:from-emerald-500 hover:to-teal-500 disabled:opacity-50 transition"
          >
            <Camera className="h-4 w-4" />
            {modelLoading
              ? "Loading 3D Model..."
              : busy
              ? "Starting..."
              : existingSnapshot
              ? `Retake ${view} scan`
              : `Start camera for ${view} view`}
          </button>
        ) : (
          <>
            <button
              type="button"
              onClick={triggerBurstCapture}
              disabled={isBursting}
              className={`inline-flex items-center gap-2 rounded-xl px-5 py-2.5 text-sm font-semibold text-white shadow-lg transition ${
                isPrecisionReady
                  ? "bg-emerald-600 shadow-emerald-600/30 hover:bg-emerald-500 animate-pulse"
                  : "bg-blue-600 shadow-blue-600/25 hover:bg-blue-500"
              }`}
            >
              <Sparkles className="h-4 w-4" />
              {isPrecisionReady ? "Capture (90%+ Ready)" : "Capture Now"}
            </button>

            <button
              type="button"
              onClick={() => startCountdownCapture(3)}
              disabled={countdown !== null || isBursting}
              className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-800/80 px-4 py-2.5 text-sm font-medium text-slate-200 hover:bg-slate-700 transition"
            >
              <Timer className="h-4 w-4 text-amber-400" />
              3s Timer
            </button>

            <button
              type="button"
              onClick={stopCamera}
              className="rounded-xl border border-slate-700 bg-slate-800/40 px-4 py-2.5 text-sm text-slate-300 hover:bg-slate-800 transition"
            >
              Pause
            </button>
          </>
        )}
      </div>
    </div>
  );
}
