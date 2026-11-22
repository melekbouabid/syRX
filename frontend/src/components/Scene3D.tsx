import { useEffect, useRef, useState } from "react";
import { LogoMark } from "./index";
import { usePrefersReducedMotion } from "./motion";
import type { SceneVariant } from "./three/pillScene";

type SceneState = "loading" | "ready" | "failed";

/**
 * What the 3D space shows. A visitor who asked for reduced motion gets the still emblem on a
 * glow and three.js is never downloaded; so does a browser without WebGL.
 */
export function SceneFrame({
  variant,
  reduced,
  state,
  className = "",
  canvasRef,
}: {
  variant: SceneVariant;
  reduced: boolean;
  state: SceneState;
  className?: string;
  canvasRef?: React.Ref<HTMLCanvasElement>;
}) {
  const still = reduced || state === "failed";
  return (
    <div
      className={`scene3d scene3d-${variant} ${still ? "scene3d-still" : `scene3d-${state}`} ${className}`}
      aria-hidden="true"
      data-motion={still ? "off" : "on"}
    >
      <span className="scene3d-glow" />
      {still ? (
        <LogoMark size={variant === "hero" ? 220 : 150} className="scene3d-emblem" />
      ) : (
        <canvas ref={canvasRef} />
      )}
    </div>
  );
}

/**
 * The animated 3D pills (three/pillScene.ts). three.js is its own chunk, imported only after
 * the page has rendered, so the first paint never waits for it. Decorative only.
 */
export function Scene3D({ variant = "hero", className = "" }: { variant?: SceneVariant; className?: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const reduced = usePrefersReducedMotion();
  const [state, setState] = useState<SceneState>("loading");

  useEffect(() => {
    if (reduced) return;
    let cancelled = false;
    let handle: { destroy(): void } | null = null;
    import("./three/pillScene")
      .then(({ createPillScene }) => {
        if (cancelled || !canvas.current) return;
        handle = createPillScene(canvas.current, variant);
        setState("ready");
      })
      .catch(() => {
        if (!cancelled) setState("failed"); // no WebGL: the still emblem instead
      });
    return () => {
      cancelled = true;
      handle?.destroy();
    };
  }, [variant, reduced]);

  return (
    <SceneFrame variant={variant} reduced={reduced} state={state} className={className} canvasRef={canvas} />
  );
}
