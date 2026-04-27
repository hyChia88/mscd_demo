import * as OBC from "@thatopen/components";
import * as THREE from "three";

// Config injected by Python into window.VIEWER_CONFIG before this module loads.
// {
//   ifc_url, target_guid, gt_guid, guid_match, static_base,
//   pool_guids, enable_click_select
// }
const cfg         = window.VIEWER_CONFIG || {};
const IFC_URL     = cfg.ifc_url     || "";
const STATIC_BASE = cfg.static_base || "";
const POOL_GUIDS  = Array.isArray(cfg.pool_guids) ? cfg.pool_guids : [];  // candidate pool
const ENABLE_CLICK_SELECT = cfg.enable_click_select === true;

let targetGuid = cfg.target_guid || "";
let gtGuid = cfg.gt_guid || "";
let guidMatch = cfg.guid_match || false;

// Colors
const COLOR_GHOST = new THREE.Color(0x475569);  // slate-600  — ghost
const COLOR_POOL  = new THREE.Color(0xf59e0b);  // amber-400  — candidate pool
const COLOR_GREEN = new THREE.Color(0x22c55e);  // green-500  — correct prediction
const COLOR_RED   = new THREE.Color(0xef4444);  // red-500    — wrong prediction
const COLOR_BLUE  = new THREE.Color(0x3b82f6);  // blue-500   — ground truth

// Worker URL — served alongside web-ifc.wasm in demo/static/
const WORKER_URL = STATIC_BASE + "/worker.mjs";
const selectableGuids = new Set(
  [...POOL_GUIDS, targetGuid, gtGuid].filter(Boolean),
);

function getPointerNdc(event, dom) {
  const rect = dom.getBoundingClientRect();
  return new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -(((event.clientY - rect.top) / rect.height) * 2 - 1),
  );
}

function isPrimaryClick(event) {
  return event.button === 0 && !event.ctrlKey && !event.metaKey && !event.shiftKey;
}

function postToParent(payload) {
  if (window.parent && window.parent !== window) {
    window.parent.postMessage(payload, "*");
  }
}


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
  const ifcLoader = components.get(OBC.IfcLoader);
  await ifcLoader.setup({ autoSetWasm: false });
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

  const model = await ifcLoader.load(flat, false, "AdvancedProject");
  world.scene.three.add(model.object);

  fragManager.core.update(performance.now());

  progressEl.style.width = "88%";
  labelEl.textContent = "Highlighting…";

  const viewerDom = world.renderer.three.domElement;
  let poolBox = null;
  let currentFocusBox = null;
  let poolLocalIds = [];
  let pointerDown = null;

  function syncFocusButton() {
    const btnFocus = document.getElementById("btn-focus");
    if (!btnFocus) return;
    const hasFocus = currentFocusBox && !currentFocusBox.isEmpty();
    btnFocus.style.display = hasFocus ? "inline-block" : "none";
  }

  async function applySelection({ focus = true } = {}) {
    await model.setColor(undefined, COLOR_GHOST);
    await model.setOpacity(undefined, 0.15);

    poolBox = null;
    if (poolLocalIds.length > 0) {
      await model.setOpacity(poolLocalIds, 0.7);
      await model.setColor(poolLocalIds, COLOR_POOL);
      try { poolBox = await model.getMergedBox(poolLocalIds); } catch (_) {}
    }

    let nextFocusBox = null;
    if (targetGuid) {
      const ids = await model.getLocalIdsByGuids([targetGuid]);
      const localId = ids[0];
      if (localId != null) {
        const targetIds = [localId];
        const color = guidMatch ? COLOR_GREEN : COLOR_RED;
        await model.resetOpacity(targetIds);
        await model.setColor(targetIds, color);
        try { nextFocusBox = await model.getMergedBox(targetIds); } catch (_) {}
      }
    }

    if (gtGuid && gtGuid !== targetGuid) {
      const gtIds = await model.getLocalIdsByGuids([gtGuid]);
      const gtLocalId = gtIds[0];
      if (gtLocalId != null) {
        await model.resetOpacity([gtLocalId]);
        await model.setColor([gtLocalId], COLOR_BLUE);
        if (!nextFocusBox) {
          try { nextFocusBox = await model.getMergedBox([gtLocalId]); } catch (_) {}
        }
      }
    }

    currentFocusBox = nextFocusBox || poolBox || null;
    syncFocusButton();

    if (focus && currentFocusBox && !currentFocusBox.isEmpty()) {
      await world.camera.controls.fitToBox(currentFocusBox, true);
    }
  }

  async function setSelection({
    targetGuid: nextTargetGuid,
    gtGuid: nextGtGuid,
    guidMatch: nextGuidMatch,
    focus = true,
  } = {}) {
    if (nextTargetGuid !== undefined) {
      targetGuid = nextTargetGuid || "";
      if (targetGuid) selectableGuids.add(targetGuid);
    }
    if (nextGtGuid !== undefined) {
      gtGuid = nextGtGuid || "";
      if (gtGuid) selectableGuids.add(gtGuid);
    }
    if (nextGuidMatch !== undefined) {
      guidMatch = !!nextGuidMatch;
    }
    await applySelection({ focus });
  }

  // ── Camera: overview → zoom to focus ─────────────────────────────────────
  // Priority: target/GT > pool-only view > nothing
  if (POOL_GUIDS.length > 0) {
    poolLocalIds = (await model.getLocalIdsByGuids(POOL_GUIDS)).filter((id) => id != null);
  }
  await applySelection({ focus: false });

  const modelBox = model.box;
  if (modelBox && !modelBox.isEmpty()) {
    await world.camera.controls.fitToBox(modelBox, false);   // instant overview
    if (currentFocusBox && !currentFocusBox.isEmpty()) {
      await world.camera.controls.fitToBox(currentFocusBox, true);  // animated zoom
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
  syncFocusButton();
  btnFocus.addEventListener("click", () => {
    if (currentFocusBox && !currentFocusBox.isEmpty()) {
      world.camera.controls.fitToBox(currentFocusBox, true);
    }
  });

  if (ENABLE_CLICK_SELECT) {
    viewerDom.addEventListener("pointerdown", (event) => {
      pointerDown = {
        x: event.clientX,
        y: event.clientY,
        t: performance.now(),
      };
    });

    viewerDom.addEventListener("pointerup", async (event) => {
      if (!isPrimaryClick(event) || !pointerDown) return;
      const moved = Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y);
      const elapsed = performance.now() - pointerDown.t;
      pointerDown = null;
      if (moved > 6 || elapsed > 700) return;

      const hit = await model.raycast({
        camera: world.camera.three,
        mouse: getPointerNdc(event, viewerDom),
        dom: viewerDom,
      });
      if (!hit || hit.localId == null) return;

      const [guid] = await model.getGuidsByLocalIds([hit.localId]);
      if (!guid || !selectableGuids.has(guid)) return;

      await setSelection({
        targetGuid: guid,
        gtGuid: guid,
        guidMatch: true,
        focus: false,
      });
      postToParent({ type: "viewer-guid-click", guid });
    });
  }

  window.addEventListener("message", (event) => {
    const data = event.data || {};
    if (data.type !== "viewer-set-selection") return;
    void setSelection({
      targetGuid: data.target_guid,
      gtGuid: data.gt_guid,
      guidMatch: data.guid_match,
      focus: data.focus !== false,
    });
  });
}

main().catch(err => {
  console.error(err);
  document.getElementById("progress-label").textContent = "Viewer error: " + err.message;
  document.getElementById("progress-bar").style.background = "#ef4444";
});
