export interface FittedGeometry {
  axis: number;
  params: number[];
  angles: number[];
  score: number;
  flat_fallback: boolean;
  nearest: number[];
  valid: boolean[];
  [field: string]: unknown;
}

export interface FittedPose {
  bank: FittedGeometry[];
  wireframe: [number, number][][];
  pose: { rvec: number[]; tvec: number[] } | null;
  bootstrap: boolean;
  message: string;
  elapsedMs: number;
  stage?: string;
  pipelineVersion?: string;
  sourceFrame?: number;
  fittingConfig?: { segment: number; fps: number; turningStartFrame: number | null;
    centers: { axis: number; side: number; color: string }[] };
}

export interface PoseObservations {
  width: number;
  height: number;
  observations: {
    points: [number, number][];
    classes: number[];
    stickers: [number, number][];
    uncertainties: number[];
  };
}
