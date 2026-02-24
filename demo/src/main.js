import * as OBC from "@thatopen/components";
import * as THREE from "three";

// Config injected by Python into window.VIEWER_CONFIG before this module loads.
// { ifc_url, target_guid, gt_guid, guid_match, static_base }
const cfg         = window.VIEWER_CONFIG || {};
const IFC_URL     = cfg.ifc_url     || "";
const TARGET_GUID = cfg.target_guid || "";
const GT_GUID     = cfg.gt_guid     || "";
const GUID_MATCH  = cfg.guid_match  || false;
const STATIC_BASE = cfg.static_base || "";

// Colors
const COLOR_GHOST = new THREE.Color(0x475569);  // slate-600  — ghost
const COLOR_GREEN = new THREE.Color(0x22c55e);  // green-500  — correct prediction
const COLOR_RED   = new THREE.Color(0xef4444);  // red-500    — wrong prediction
const COLOR_BLUE  = new THREE.Color(0x3b82f6);  // blue-500   — ground truth

// Worker URL — served alongside web-ifc.wasm in demo/static/
// NOTE: STATIC_BASE is already ".../demo/static", so worker is at STATIC_BASE + "/worker.mjs"
const WORKER_URL = STATIC_BASE + "/worker.mjs";


async function main() {
  const container  = document.getElementById("viewer");
  const progressEl = document.getElementById("progress-bar");
  const labelEl    = document.getElementById("progress-label");

  // ── Scene setup ─────────────────────────────────────────────────────────
  const components = new OBC.Components();
  const worlds     = components.get(OBC.Worlds);
  const world      = worlds.create();

  world.scene    = new OBC.SimpleScene(components);
  world.renderer = new OBC.SimpleRenderer(components, container);
  world.camera   = new OBC.SimpleCamera(components);

  world.scene.setup();
  components.init();

  // ── Initialize FragmentsManager (required before ifcLoader.load) ─────────
  const fragManager = components.get(OBC.FragmentsManager);
  fragManager.init(WORKER_URL);

  // Drive tile rendering every frame
  world.onAfterUpdate.add(() => {
    fragManager.core.update(performance.now());
  });

  // ── IFC loader ──────────────────────────────────────────────────────────
  // autoSetWasm: false — skip unpkg.com network fetch (hangs in offline/local envs)
  const ifcLoader = components.get(OBC.IfcLoader);
  await ifcLoader.setup({ autoSetWasm: false });
  // STATIC_BASE = "http://localhost:8502/demo/static" → wasm at STATIC_BASE + "/"
  ifcLoader.settings.wasm = { path: STATIC_BASE + "/", absolute: true };

  // ── Stream download with progress bar ───────────────────────────────────
  labelEl.textContent = "Fetching IFC…";
  progressEl.style.width = "5%";
  const response = await fetch(IFC_URL);
  const total    = parseInt(response.headers.get("content-length") || "0");
  const reader   = response.body.getReader();
  const chunks   = [];
  let received   = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total > 0) progressEl.style.width = (5 + Math.round((received / total) * 70)) + "%";
  }

  labelEl.textContent = "Parsing geometry…";
  progressEl.style.width = "77%";

  const flat = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { flat.set(c, off); off += c.length; }

  // load(data, coordinate=false, name) — coordinate=false preserves IFC coords
  const model = await ifcLoader.load(flat, false, "AdvancedProject");
  world.scene.three.add(model.object);

  // Initial tile update after model is added to scene
  fragManager.core.update(performance.now());

  progressEl.style.width = "88%";
  labelEl.textContent = "Highlighting…";

  // ── Ghost mode: paint all elements gray + semi-transparent ───────────────
  await model.setColor(undefined, COLOR_GHOST);
  await model.setOpacity(undefined, 0.15);

  // ── Highlight predicted element ──────────────────────────────────────────
  let targetBox = null;
  if (TARGET_GUID) {
    const ids = await model.getLocalIdsByGuids([TARGET_GUID]);
    const localId = ids[0];
    if (localId != null) {
      const targetIds = [localId];
      const color = GUID_MATCH ? COLOR_GREEN : COLOR_RED;
      await model.resetOpacity(targetIds);
      await model.setColor(targetIds, color);
      try { targetBox = await model.getMergedBox(targetIds); } catch (_) {}
    }
  }

  // ── Highlight GT in blue when prediction differs ─────────────────────────
  if (GT_GUID && GT_GUID !== TARGET_GUID) {
    const gtIds = await model.getLocalIdsByGuids([GT_GUID]);
    const gtLocalId = gtIds[0];
    if (gtLocalId != null) {
      await model.resetOpacity([gtLocalId]);
      await model.setColor([gtLocalId], COLOR_BLUE);
      if (!targetBox) {
        try { targetBox = await model.getMergedBox([gtLocalId]); } catch (_) {}
      }
    }
  }

  // ── Camera: instant overview → animated zoom to target ───────────────────
  const modelBox = model.box;
  if (modelBox && !modelBox.isEmpty()) {
    await world.camera.controls.fitToBox(modelBox, false);   // instant overview
    if (targetBox && !targetBox.isEmpty()) {
      await world.camera.controls.fitToBox(targetBox, true); // animated zoom
    }
  }

  // ── Finish progress ──────────────────────────────────────────────────────
  progressEl.style.width = "100%";
  labelEl.textContent = "Ready";
  setTimeout(() => { document.getElementById("progress-wrap").style.display = "none"; }, 600);

  // ── Toolbar ──────────────────────────────────────────────────────────────
  const toolbar = document.getElementById("toolbar");
  if (toolbar) toolbar.style.display = "flex";

  document.getElementById("btn-fit-all").addEventListener("click", () => {
    const box = model.box;
    if (box && !box.isEmpty()) world.camera.controls.fitToBox(box, true);
  });

  const btnFocus = document.getElementById("btn-focus");
  if (targetBox && !targetBox.isEmpty()) {
    btnFocus.style.display = "inline-block";
    btnFocus.addEventListener("click", () => {
      world.camera.controls.fitToBox(targetBox, true);
    });
  }
}

main().catch(err => {
  console.error(err);
  document.getElementById("progress-label").textContent = "Viewer error: " + err.message;
  document.getElementById("progress-bar").style.background = "#ef4444";
});
