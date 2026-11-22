// The animated 3D scene of the start page and the sign-in page (three.js), loaded only on
// those pages and only when the visitor accepts motion (components/Scene3D.tsx).
//
// It shows what Tujya does: a big glossy capsule in the logo's colours, a detection box and a
// scan line sweeping it (the vision step), small capsules and tablets orbiting it, and glowing
// links with pulses running between them (the interaction check). Everything is built from
// code: no model or texture is downloaded.
//
// Cost control: frames are drawn only while the canvas is on screen and the tab is visible,
// the pixel ratio is capped at 2, and destroy() frees every GPU resource.

import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

export type SceneVariant = "hero" | "compact";

export interface SceneHandle {
  destroy(): void;
}

const COLOURS = {
  navy: "#0a3558",
  blue: "#145788",
  sky: "#5fb0e6",
  teal: "#2f9ba3",
  green: "#87ba55",
  lime: "#b9de8a",
  white: "#f3f6f9",
  cream: "#f6efd9",
  amber: "#f2b544",
};

// ------------------------------------------------------------------ shapes

/** Half of a capsule as a lathe profile: a cylinder from the seam, closed by a dome. */
function capsuleHalf(radius: number, length: number, flip: boolean): THREE.LatheGeometry {
  const points = [new THREE.Vector2(radius, 0), new THREE.Vector2(radius, length / 2)];
  for (let i = 1; i <= 14; i += 1) {
    const angle = (i / 14) * (Math.PI / 2);
    points.push(
      new THREE.Vector2(Math.cos(angle) * radius, length / 2 + Math.sin(angle) * radius),
    );
  }
  points[points.length - 1].x = 0.0001;
  if (flip) {
    points.forEach((point) => (point.y = -point.y));
    points.reverse(); // keep the faces pointing outwards
  }
  return new THREE.LatheGeometry(points, 48);
}

/** A round tablet with domed faces and a rounded edge. */
function tabletGeometry(radius: number, thickness: number): THREE.LatheGeometry {
  const edge = thickness / 2;
  const dome = thickness * 0.22;
  const flat = radius - edge;
  const points: THREE.Vector2[] = [];
  for (let i = 0; i <= 10; i += 1) {
    const x = (i / 10) * flat;
    points.push(new THREE.Vector2(x, edge + dome * (1 - (x / flat) ** 2)));
  }
  for (let i = 1; i < 10; i += 1) {
    const angle = Math.PI / 2 - (i / 10) * Math.PI;
    points.push(new THREE.Vector2(flat + Math.cos(angle) * edge, Math.sin(angle) * edge));
  }
  for (let i = 10; i >= 0; i -= 1) {
    const x = (i / 10) * flat;
    points.push(new THREE.Vector2(x, -edge - dome * (1 - (x / flat) ** 2)));
  }
  points[0].x = 0.0001;
  points[points.length - 1].x = 0.0001;
  return new THREE.LatheGeometry(points.reverse(), 56);
}

function glossy(colour: string): THREE.MeshPhysicalMaterial {
  return new THREE.MeshPhysicalMaterial({
    color: colour,
    roughness: 0.22,
    clearcoat: 1,
    clearcoatRoughness: 0.12,
  });
}

function chalky(colour: string): THREE.MeshPhysicalMaterial {
  return new THREE.MeshPhysicalMaterial({
    color: colour,
    roughness: 0.62,
    clearcoat: 0.35,
    clearcoatRoughness: 0.4,
  });
}

function capsule(radius: number, length: number, top: string, bottom: string): THREE.Group {
  const group = new THREE.Group();
  const gap = length * 0.06; // the cap slides over the body, like a real capsule
  const upper = new THREE.Mesh(capsuleHalf(radius * 1.035, length, false), glossy(top));
  upper.position.y = -gap;
  const lower = new THREE.Mesh(capsuleHalf(radius, length, true), glossy(bottom));
  lower.position.y = gap;
  group.add(upper, lower);
  return group;
}

function tablet(radius: number, colour: string): THREE.Group {
  const group = new THREE.Group();
  const body = new THREE.Mesh(tabletGeometry(radius, radius * 0.42), chalky(colour));
  const score = new THREE.Mesh(
    new THREE.BoxGeometry(radius * 1.5, radius * 0.03, radius * 0.06),
    new THREE.MeshStandardMaterial({ color: "#c9d2db", roughness: 0.8 }),
  );
  score.position.y = radius * 0.3;
  group.add(body, score);
  return group;
}

/** A soft round dot drawn once on a small canvas: particles and pulses. */
function dotTexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 64;
  const context = canvas.getContext("2d");
  if (context) {
    const gradient = context.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, "rgba(255,255,255,1)");
    gradient.addColorStop(0.35, "rgba(255,255,255,0.55)");
    gradient.addColorStop(1, "rgba(255,255,255,0)");
    context.fillStyle = gradient;
    context.fillRect(0, 0, 64, 64);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

/** Four L-shaped corners around a w x h box: the detector's bounding box. */
function bracketGeometry(w: number, h: number, arm: number): THREE.BufferGeometry {
  const x = w / 2;
  const y = h / 2;
  const corners = [
    [-x, y - arm, -x, y, -x + arm, y],
    [x - arm, y, x, y, x, y - arm],
    [x, -y + arm, x, -y, x - arm, -y],
    [-x + arm, -y, -x, -y, -x, -y + arm],
  ];
  const positions: number[] = [];
  for (const [ax, ay, bx, by, cx, cy] of corners) {
    positions.push(ax, ay, 0, bx, by, 0, bx, by, 0, cx, cy, 0);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  return geometry;
}

// ------------------------------------------------------------------ scene

interface Orbiter {
  object: THREE.Object3D;
  angle: number;
  radius: number;
  speed: number;
  depth: number;
  bob: number;
  phase: number;
  spin: THREE.Vector3;
}

export function createPillScene(canvas: HTMLCanvasElement, variant: SceneVariant): SceneHandle {
  const host = canvas.parentElement ?? canvas;
  const compact = variant === "compact";

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  const environment = pmrem.fromScene(room, 0.04).texture;
  scene.environment = environment;

  const cameraZ = compact ? 11 : 14;
  const camera = new THREE.PerspectiveCamera(compact ? 38 : 34, 1, 0.1, 100);
  camera.position.set(0, 0, cameraZ);

  // A cool key light, a green rim and a blue fill: the gloss picks up the logo's colours.
  scene.add(new THREE.HemisphereLight("#dfefff", "#0b2236", 0.7));
  const key = new THREE.DirectionalLight("#ffffff", 1.6);
  key.position.set(4, 6, 8);
  const rim = new THREE.PointLight(COLOURS.green, 40, 30);
  rim.position.set(-6, -3, 4);
  const fill = new THREE.PointLight(COLOURS.sky, 50, 30);
  fill.position.set(6, 2, -2);
  scene.add(key, rim, fill);

  const world = new THREE.Group();
  scene.add(world);
  const dot = dotTexture();

  // The big capsule, with the detector's box and a scan line over it.
  const hero = new THREE.Group();
  world.add(hero);
  const big = capsule(0.82, 2.5, COLOURS.blue, COLOURS.green);
  big.rotation.z = -0.55;
  hero.add(big);

  const boxW = 3.1;
  const boxH = 3.6;
  const brackets = new THREE.LineSegments(
    bracketGeometry(boxW, boxH, 0.55),
    new THREE.LineBasicMaterial({ color: COLOURS.teal, transparent: true }),
  );
  brackets.position.z = 1.2;
  hero.add(brackets);

  const scan = new THREE.Mesh(
    new THREE.PlaneGeometry(boxW * 1.02, 0.06),
    new THREE.MeshBasicMaterial({
      color: "#9fe8e2",
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );
  scan.position.z = 1.25;
  hero.add(scan);

  const ringMaterial = new THREE.MeshBasicMaterial({
    color: COLOURS.sky,
    transparent: true,
    opacity: 0.5,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const rings = [2.35, 2.75].map((radius, index) => {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(radius, 0.012, 8, 200), ringMaterial);
    ring.rotation.set(1.15 + index * 0.35, 0.2 - index * 0.5, 0);
    hero.add(ring);
    return ring;
  });

  // Small pills orbiting the capsule. A fixed seed: the same layout on every visit.
  let seed = 7;
  const random = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
  const kinds: [string, string, string?][] = [
    ["capsule", COLOURS.white, COLOURS.blue],
    ["tablet", COLOURS.white],
    ["capsule", COLOURS.green, COLOURS.white],
    ["tablet", COLOURS.cream],
    ["capsule", COLOURS.amber, COLOURS.white],
    ["tablet", "#dbeefa"],
    ["capsule", COLOURS.navy, COLOURS.sky],
    ["tablet", COLOURS.lime],
  ];
  const used = compact ? kinds.slice(0, 5) : kinds;
  const orbiters: Orbiter[] = used.map(([kind, a, b], index) => {
    const object =
      kind === "capsule"
        ? capsule(0.26 + random() * 0.08, 0.75 + random() * 0.25, a, b ?? COLOURS.white)
        : tablet(0.36 + random() * 0.1, a);
    object.rotation.set(random() * 3, random() * 3, random() * 3);
    world.add(object);
    return {
      object,
      angle: (index / used.length) * Math.PI * 2 + random() * 0.4,
      radius: (compact ? 2.9 : 3.6) + random() * (compact ? 0.6 : 1.2),
      speed: 0.045 + random() * 0.03,
      depth: -0.6 - random() * 2.4,
      bob: 0.12 + random() * 0.18,
      phase: random() * Math.PI * 2,
      spin: new THREE.Vector3(random() - 0.5, random() * 0.6 - 0.3, random() * 0.4 - 0.2),
    };
  });

  // Interaction links between pills (-1 is the big capsule), with a pulse running along each.
  const pairs: [number, number][] = compact
    ? [
        [-1, 0],
        [-1, 2],
        [1, 3],
        [-1, 4],
      ]
    : [
        [-1, 0],
        [-1, 2],
        [-1, 5],
        [1, 3],
        [4, 6],
        [-1, 7],
        [2, 4],
      ];
  const linkPositions = new Float32Array(pairs.length * 6);
  const linkGeometry = new THREE.BufferGeometry();
  linkGeometry.setAttribute("position", new THREE.BufferAttribute(linkPositions, 3));
  const links = new THREE.LineSegments(
    linkGeometry,
    new THREE.LineBasicMaterial({
      color: COLOURS.teal,
      transparent: true,
      opacity: 0.35,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );
  world.add(links);
  const pulsePositions = new Float32Array(pairs.length * 3);
  const pulseGeometry = new THREE.BufferGeometry();
  pulseGeometry.setAttribute("position", new THREE.BufferAttribute(pulsePositions, 3));
  const pulses = new THREE.Points(
    pulseGeometry,
    new THREE.PointsMaterial({
      map: dot,
      color: COLOURS.lime,
      size: 0.34,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );
  world.add(pulses);

  // Faint particles far behind.
  const count = compact ? 200 : 460;
  const particlePositions = new Float32Array(count * 3);
  const particleColours = new Float32Array(count * 3);
  const tones = [COLOURS.sky, COLOURS.green, "#ffffff"].map((c) => new THREE.Color(c));
  for (let i = 0; i < count; i += 1) {
    const r = 6 + random() * 9;
    const theta = random() * Math.PI * 2;
    const phi = Math.acos(2 * random() - 1);
    particlePositions.set(
      [
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.cos(phi) * 0.7,
        -Math.abs(r * Math.sin(phi) * Math.sin(theta)) - 2,
      ],
      i * 3,
    );
    const tone = tones[i % 3];
    particleColours.set([tone.r, tone.g, tone.b], i * 3);
  }
  const particleGeometry = new THREE.BufferGeometry();
  particleGeometry.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3));
  particleGeometry.setAttribute("color", new THREE.BufferAttribute(particleColours, 3));
  const particles = new THREE.Points(
    particleGeometry,
    new THREE.PointsMaterial({
      map: dot,
      size: 0.11,
      vertexColors: true,
      transparent: true,
      opacity: 0.7,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );
  scene.add(particles);

  // ---------------------------------------------------------------- layout and pointer

  let squeeze = 1; // wide hero: pills on the text side keep closer to the capsule
  const layout = () => {
    const rect = host.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    const wide = !compact && camera.aspect > 1.15;
    squeeze = wide ? 0.34 : 1;
    world.position.x = wide ? 3.4 : 0;
    const fit = compact ? camera.aspect / 0.95 : wide ? 1 : camera.aspect / 0.8;
    world.scale.setScalar(Math.max(0.62, Math.min(1, fit)));
  };
  layout();

  const pointer = { x: 0, y: 0 };
  const onPointer = (event: PointerEvent) => {
    pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
    pointer.y = (event.clientY / window.innerHeight) * 2 - 1;
  };
  window.addEventListener("pointermove", onPointer, { passive: true });

  // ---------------------------------------------------------------- animation

  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const started = performance.now();
  const centre = (index: number) => (index < 0 ? hero : orbiters[index].object);

  const update = (t: number) => {
    const grow = (delay: number) => {
      const p = Math.min(1, Math.max(0, (t - delay) / 0.9));
      return p <= 0 ? 0.0001 : 1 + 2.5 * (p - 1) ** 3 + 1.5 * (p - 1) ** 2; // ease out, back
    };
    hero.scale.setScalar(grow(0));
    hero.position.y = Math.sin(t * 0.9) * 0.12;
    big.rotation.y = t * 0.35;
    big.rotation.x = Math.sin(t * 0.4) * 0.12;

    orbiters.forEach((orbiter, index) => {
      const angle = orbiter.angle + t * orbiter.speed;
      const x = Math.cos(angle) * orbiter.radius;
      orbiter.object.position.set(
        x < 0 ? x * squeeze : x,
        Math.sin(angle) * orbiter.radius * 0.78 + Math.sin(t * 0.9 + orbiter.phase) * orbiter.bob,
        orbiter.depth,
      );
      orbiter.object.scale.setScalar(grow(0.25 + index * 0.07));
      orbiter.object.rotation.x += orbiter.spin.x * 0.01;
      orbiter.object.rotation.y += orbiter.spin.y * 0.01;
      orbiter.object.rotation.z += orbiter.spin.z * 0.01;
    });

    rings[0].rotation.z = t * 0.25;
    rings[1].rotation.z = -t * 0.18;
    scan.position.y = Math.sin(t * 1.4) * (boxH / 2 - 0.25);
    const reveal = Math.min(1, Math.max(0, (t - 0.6) / 0.6));
    const breathe = 1 + Math.sin(t * 2.2) * 0.018;
    brackets.scale.set(breathe, breathe, 1);
    (brackets.material as THREE.LineBasicMaterial).opacity = 0.95 * reveal;
    (scan.material as THREE.MeshBasicMaterial).opacity =
      (0.5 + 0.4 * Math.abs(Math.cos(t * 1.4))) * reveal;

    pairs.forEach(([i, j], index) => {
      centre(i).getWorldPosition(a);
      centre(j).getWorldPosition(b);
      world.worldToLocal(a);
      world.worldToLocal(b);
      linkPositions.set([a.x, a.y, a.z, b.x, b.y, b.z], index * 6);
      const along = (t * 0.32 + index * 0.37) % 1;
      pulsePositions.set(
        [a.x + (b.x - a.x) * along, a.y + (b.y - a.y) * along, a.z + (b.z - a.z) * along],
        index * 3,
      );
    });
    linkGeometry.attributes.position.needsUpdate = true;
    pulseGeometry.attributes.position.needsUpdate = true;
    (links.material as THREE.LineBasicMaterial).opacity = (0.22 + 0.12 * Math.sin(t * 1.7)) * reveal;

    particles.rotation.y = t * 0.02;
    world.rotation.y = Math.sin(t * 0.15) * 0.12 + pointer.x * 0.18;
    world.rotation.x = pointer.y * 0.1;
    camera.position.x += (pointer.x * 0.6 - camera.position.x) * 0.05;
    camera.position.y += (-pointer.y * 0.4 - camera.position.y) * 0.05;
    camera.lookAt(world.position.x * 0.35, 0, 0);
  };

  let frame = 0;
  let running = false;
  let onScreen = true;
  const draw = () => {
    update((performance.now() - started) / 1000);
    renderer.render(scene, camera);
  };
  const loop = () => {
    frame = requestAnimationFrame(loop);
    draw();
  };
  const start = () => {
    if (running || !onScreen || document.hidden) return;
    running = true;
    frame = requestAnimationFrame(loop);
  };
  const stop = () => {
    running = false;
    cancelAnimationFrame(frame);
  };

  const resizeObserver = new ResizeObserver(() => {
    layout();
    if (!running) draw();
  });
  resizeObserver.observe(host);
  const visibility = new IntersectionObserver(([entry]) => {
    onScreen = entry.isIntersecting;
    if (onScreen) start();
    else stop();
  });
  visibility.observe(host);
  const onVisibility = () => (document.hidden ? stop() : start());
  document.addEventListener("visibilitychange", onVisibility);

  draw();
  start();

  return {
    destroy() {
      stop();
      resizeObserver.disconnect();
      visibility.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("pointermove", onPointer);
      scene.traverse((object) => {
        const mesh = object as THREE.Mesh;
        mesh.geometry?.dispose();
        const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
        for (const item of Array.isArray(material) ? material : material ? [material] : []) {
          item.dispose();
        }
      });
      dot.dispose();
      environment.dispose();
      room.dispose();
      pmrem.dispose();
      renderer.dispose();
    },
  };
}
