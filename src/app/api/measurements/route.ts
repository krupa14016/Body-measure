import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { measurementKeys } from "@/lib/measurement";

const submissionSchema = z.object({
  clientName: z.string().max(100).optional(),
  heightCm: z.number().min(120).max(230),
  unit: z.enum(["cm", "in"]).default("cm"),
  genderPreference: z.enum(["unisex", "male", "female"]).default("unisex"),
  modelVersion: z.string().default("tailor-ai-1.0.0-lite"),
  measurements: z.record(z.string(), z.number().nullable()),
});

// In-memory store fallback for development before external PostgreSQL connection is configured
interface StoredSubmission {
  id: string;
  clientName?: string;
  heightCm: number;
  unit: "cm" | "in";
  genderPreference: string;
  modelVersion: string;
  measurements: Record<string, number | null>;
  createdAt: string;
}

const memoryStore: StoredSubmission[] = [];

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const parsed = submissionSchema.safeParse(body);

    if (!parsed.success) {
      return NextResponse.json(
        { error: "Validation failed", issues: parsed.error.issues },
        { status: 400 }
      );
    }

    const { clientName, heightCm, unit, genderPreference, modelVersion, measurements } = parsed.data;

    const submission: StoredSubmission = {
      id: "sub_" + Math.random().toString(36).substring(2, 10),
      clientName,
      heightCm,
      unit,
      genderPreference,
      modelVersion,
      measurements,
      createdAt: new Date().toISOString(),
    };

    memoryStore.unshift(submission);

    return NextResponse.json({
      success: true,
      message: "Measurement record saved successfully",
      submission,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Internal server error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json({
    total: memoryStore.length,
    submissions: memoryStore,
  });
}
