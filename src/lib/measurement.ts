import { z } from "zod";

export const profileSchema = z.object({
  heightCm: z.number().min(120, "Enter a height of at least 120 cm").max(230, "Enter a height of at most 230 cm"),
  weightKg: z.number().min(30, "Enter weight of at least 30 kg").max(200, "Enter weight of at most 200 kg").optional(),
  preferredUnit: z.enum(["cm", "in"]).default("cm"),
  genderPreference: z.enum(["unisex", "male", "female"]).default("unisex"),
  bodyBuild: z.enum(["slim", "regular", "athletic", "heavy"]).default("regular"),
  clientName: z.string().max(60).optional(),
});

export type ProfileData = z.infer<typeof profileSchema>;

export const measurementKeys = [
  "shoulderWidth",
  "chestCircumference",
  "waistCircumference",
  "hipCircumference",
  "neckCircumference",
  "sleeveLength",
  "armLength",
  "shirtLength",
  "trouserWaist",
  "inseam",
  "outseam",
  "thighCircumference",
] as const;

export type MeasurementKey = typeof measurementKeys[number];

export type MeasurementValues = Record<MeasurementKey, number | null>;

export interface MeasurementMetadata {
  label: string;
  category: "upper" | "lower" | "circumference" | "length";
  description: string;
  tailoringTip: string;
  benchmarkMAE: string; // Mean Absolute Error benchmark against physical tape
  accuracyGrade: string; // Expected accuracy range
  isCircumference: boolean;
}

export const measurementMetadata: Record<MeasurementKey, MeasurementMetadata> = {
  shoulderWidth: {
    label: "Shoulder width",
    category: "upper",
    description: "Distance across the back from the tip of one acromion (shoulder bone) to the other.",
    tailoringTip: "Crucial for jacket and shirt yoke fit. Key for structural tailoring.",
    benchmarkMAE: "± 0.9 cm",
    accuracyGrade: "94–96%",
    isCircumference: false,
  },
  chestCircumference: {
    label: "Chest circumference",
    category: "circumference",
    description: "Around the fullest part of the chest, under armpits and across shoulder blades.",
    tailoringTip: "Keep tape horizontal and breathe normally.",
    benchmarkMAE: "± 1.4 cm",
    accuracyGrade: "91–93%",
    isCircumference: true,
  },
  waistCircumference: {
    label: "Waist circumference",
    category: "circumference",
    description: "Around the natural waistline, typically narrowest point above navel.",
    tailoringTip: "Keep one finger between tape and body for comfortable ease.",
    benchmarkMAE: "± 1.3 cm",
    accuracyGrade: "91–93%",
    isCircumference: true,
  },
  hipCircumference: {
    label: "Hip circumference",
    category: "circumference",
    description: "Around the fullest part of hips and buttocks with feet together.",
    tailoringTip: "Ensure tape is level all the way around.",
    benchmarkMAE: "± 1.5 cm",
    accuracyGrade: "90–92%",
    isCircumference: true,
  },
  neckCircumference: {
    label: "Neck circumference",
    category: "circumference",
    description: "Around base of neck where collar sits, with one finger allowance.",
    tailoringTip: "Essential for dress shirt collar sizes.",
    benchmarkMAE: "± 0.8 cm",
    accuracyGrade: "92–94%",
    isCircumference: true,
  },
  sleeveLength: {
    label: "Sleeve length",
    category: "upper",
    description: "From shoulder point down over the slightly bent elbow to the wrist bone.",
    tailoringTip: "Standard dress shirts end just past the wrist bone.",
    benchmarkMAE: "± 1.1 cm",
    accuracyGrade: "93–95%",
    isCircumference: false,
  },
  armLength: {
    label: "Arm length",
    category: "upper",
    description: "Direct length from shoulder point to the wrist.",
    tailoringTip: "Used for jackets, t-shirts, and knitwear sizing.",
    benchmarkMAE: "± 1.0 cm",
    accuracyGrade: "94–96%",
    isCircumference: false,
  },
  shirtLength: {
    label: "Shirt length",
    category: "length",
    description: "From base of neck / cervicale down past hips to desired hemline.",
    tailoringTip: "Tucked shirts are usually 4-7 cm longer than untucked styles.",
    benchmarkMAE: "± 1.3 cm",
    accuracyGrade: "92–94%",
    isCircumference: false,
  },
  trouserWaist: {
    label: "Trouser waist",
    category: "lower",
    description: "Where trousers are naturally worn (usually slightly lower than natural waist).",
    tailoringTip: "Check against user's preferred rise (low, mid, or high rise).",
    benchmarkMAE: "± 1.4 cm",
    accuracyGrade: "91–93%",
    isCircumference: true,
  },
  inseam: {
    label: "Inseam",
    category: "lower",
    description: "From highest point of crotch down inner leg to bottom of ankle / shoe top.",
    tailoringTip: "Best measured wearing the type of shoes intended with trousers.",
    benchmarkMAE: "± 1.2 cm",
    accuracyGrade: "93–95%",
    isCircumference: false,
  },
  outseam: {
    label: "Outseam",
    category: "lower",
    description: "From top of waistband down the outside of the leg to the shoe.",
    tailoringTip: "Combined with inseam to determine trouser rise.",
    benchmarkMAE: "± 1.3 cm",
    accuracyGrade: "92–94%",
    isCircumference: false,
  },
  thighCircumference: {
    label: "Thigh circumference",
    category: "circumference",
    description: "Around the fullest part of upper thigh, just below crotch level.",
    tailoringTip: "Ensures slim fit trousers are not restrictive when sitting.",
    benchmarkMAE: "± 1.4 cm",
    accuracyGrade: "90–92%",
    isCircumference: true,
  },
};

export const measurementLabels: Record<MeasurementKey, string> = Object.fromEntries(
  measurementKeys.map((key) => [key, measurementMetadata[key].label])
) as Record<MeasurementKey, string>;

export function cmToInches(cm: number): number {
  return cm / 2.54;
}

export function inchesToCm(inches: number): number {
  return inches * 2.54;
}

export function kgToLbs(kg: number): number {
  return kg * 2.20462;
}

export function lbsToKg(lbs: number): number {
  return lbs / 2.20462;
}

export function displayCm(cm: number | null, unit: "cm" | "in"): string {
  if (cm === null || isNaN(cm)) return "—";
  return unit === "cm" ? cm.toFixed(1) : cmToInches(cm).toFixed(1);
}

export interface SavedDraft {
  id: string;
  clientName?: string;
  heightCm: number;
  weightKg?: number;
  unit: "cm" | "in";
  genderPreference: "unisex" | "male" | "female";
  bodyBuild?: "slim" | "regular" | "athletic" | "heavy";
  measurements: MeasurementValues;
  calibrationFactor?: number;
  estimatesMetadata?: Record<string, {
    quality: "good" | "review" | "retake" | "manual";
    method: string;
    confidence: number;
    notes?: string;
  }>;
  frontSnapshotUrl?: string;
  sideSnapshotUrl?: string;
  savedAt: string;
}
