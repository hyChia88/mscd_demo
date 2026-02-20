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

async function main() {
  const container  = document.getElementById("viewer");
  const progressEl = document.getElementById("progress-bar");
  const labelEl    = document.getElementById("progress-label");

  // scene
  const components = new OBC.Components();
  const worlds     = components.get(OBC.Worlds);
  const world      = worlds.create();

  world.scene    = new OBC.SimpleScene(components);
  world.renderer = new OBC.SimpleRenderer(components, container);
  world.camera   = new OBC.SimpleCamera(components);

  world.scene.setup();
  components.init();
  world.camera.controls.setLookAt(30, 20, 30, 0, 5, 0);

  // IFC loader
  const ifcLoader = components.get(OBC.IfcLoader);
  await ifcLoader.setup();
  ifcLoader.settings.wasm = { path: STATIC_BASE + "/", absolute: false };

  // stream load with progress
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
    if (total > 0) progressEl.style.width = (5 + Math.round((received / total) * 75)) + "%";
  }

  labelEl.textContent = "Parsing geometry…";
  progressEl.style.width = "82%";

  const flat = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { flat.set(c, off); off += c.length; }

  const model = await ifcLoader.load(flat);
  world.scene.three.add(model);

  progressEl.style.width = "100%";
  labelEl.textContent = "Ready";
  setTimeout(() => { document.getElementById("progress-wrap").style.display = "none"; }, 600);

  // highlight predicted element
  if (TARGET_GUID) {
    const localIds = await model.getLocalIdsByGuids([TARGET_GUID]);
    if (localIds && localIds.size > 0) {
      const color = GUID_MATCH
        ? new THREE.Color(0x22c55e)   // green — correct
        : new THREE.Color(0xef4444);  // red   — wrong
      await model.highlight("target", localIds, { color, opacity: 1.0, renderedFaces: "all" });
    }
  }

  // highlight GT in blue when different from prediction
  if (GT_GUID && GT_GUID !== TARGET_GUID) {
    const gtIds = await model.getLocalIdsByGuids([GT_GUID]);
    if (gtIds && gtIds.size > 0) {
      await model.highlight("gt", gtIds, { color: new THREE.Color(0x3b82f6), opacity: 1.0, renderedFaces: "all" });
    }
  }
}

main().catch(err => {
  console.error(err);
  document.getElementById("progress-label").textContent = "Viewer error: " + err.message;
  document.getElementById("progress-bar").style.background = "#ef4444";
});
