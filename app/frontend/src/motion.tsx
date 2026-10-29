// Motion helpers. Everything here falls back to "just show it" when the visitor has asked
// their system for reduced motion, or when the browser cannot observe scrolling.

import { useEffect, useRef, useState, type ReactNode, type RefObject } from "react";

const numberFormat = new Intl.NumberFormat("en");

export function usePrefersReducedMotion(): boolean {
  const query = "(prefers-reduced-motion: reduce)";
  const [reduced, setReduced] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const media = window.matchMedia(query);
    const update = () => setReduced(media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  return reduced;
}

/** True once the element has scrolled into view. It never goes back to false. */
export function useInView(ref: RefObject<Element | null>): boolean {
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const node = ref.current;
    if (!node || inView) return;
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          observer.disconnect();
        }
      },
      { rootMargin: "0px 0px -6% 0px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [ref, inView]);
  return inView;
}

interface RevealProps {
  children: ReactNode;
  className?: string;
  /** Animate the direct children one after the other instead of the block as a whole. */
  stagger?: boolean;
}

/** Fade and slide its content in the first time it is scrolled into view. */
export function Reveal({ children, className = "", stagger = false }: RevealProps) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref);
  const kind = stagger ? "stagger" : "reveal";
  return (
    <div ref={ref} className={`${kind} ${inView ? "is-visible" : ""} ${className}`}>
      {children}
    </div>
  );
}

interface CountUpProps {
  value: number;
  /** Milliseconds for the whole count. */
  duration?: number;
}

/**
 * A number that counts up to its value when it first appears, and from its previous value
 * when the value changes. `data-value` always holds the final figure.
 */
export function CountUp({ value, duration = 900 }: CountUpProps) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref);
  const reduced = usePrefersReducedMotion();
  const [shown, setShown] = useState(reduced ? value : 0);
  const current = useRef(shown);
  current.current = shown;

  useEffect(() => {
    if (!inView) return;
    const from = current.current;
    if (reduced || from === value) {
      setShown(value);
      return;
    }
    const start = performance.now();
    let frame = requestAnimationFrame(function tick(now: number) {
      const progress = Math.min(1, (now - start) / duration);
      const eased = 1 - (1 - progress) ** 3;
      setShown(Math.round(from + (value - from) * eased));
      if (progress < 1) frame = requestAnimationFrame(tick);
    });
    return () => cancelAnimationFrame(frame);
  }, [inView, value, duration, reduced]);

  return (
    <span ref={ref} className="count-up" data-value={numberFormat.format(value)}>
      {numberFormat.format(shown)}
    </span>
  );
}
