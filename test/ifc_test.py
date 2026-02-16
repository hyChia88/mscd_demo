#!/usr/bin/env python3
"""
IFC Processor for Research Prototype
Extracts, filters, visualizes and exports IFC building elements
Part of: Integrated Agentic Interpreter for Evidence-Linked and Compliance-Ready BIM Data
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import ifcopenshell
import ifcopenshell.util.element
import ifcopenshell.geom


class IFCProcessor:
    """Main processor for IFC data extraction and export"""
    
    # Real building element types to extract
    BUILDING_ELEMENT_TYPES = [
        "IfcWall", "IfcWallStandardCase",
        "IfcSlab", "IfcDoor", "IfcWindow",
        "IfcColumn", "IfcBeam", "IfcStair",
        "IfcCovering", "IfcRailing", "IfcRoof",
        "IfcFooting", "IfcPile", "IfcCurtainWall",
        "IfcPlate", "IfcMember", "IfcRamp"
    ]
    
    def __init__(self, ifc_path: str):
        """Initialize with IFC file path"""
        self.ifc_path = Path(ifc_path)
        if not self.ifc_path.exists():
            raise FileNotFoundError(f"IFC file not found: {ifc_path}")
        
        print(f"Loading IFC file: {self.ifc_path}")
        self.model = ifcopenshell.open(str(self.ifc_path))
        print(f"Schema: {self.model.schema}")
    
    def extract_all_entities(self) -> List[Dict[str, Any]]:
        """Extract all entities from IFC model"""
        items = []
        for e in self.model:
            ent = {
                "id": e.id(),
                "type": e.is_a(),
                "name": getattr(e, "Name", None),
                "global_id": getattr(e, "GlobalId", None),
            }
            items.append(ent)
        return items
    
    def extract_building_elements(self, include_psets: bool = True) -> List[Dict[str, Any]]:
        """
        Extract only real building elements
        
        Args:
            include_psets: Whether to include property sets (slower but more complete)
        
        Returns:
            List of building element dictionaries
        """
        building_elems = []
        
        for elem_type in self.BUILDING_ELEMENT_TYPES:
            try:
                for el in self.model.by_type(elem_type):
                    data = {
                        "id": el.id(),
                        "type": el.is_a(),
                        "name": getattr(el, "Name", None),
                        "global_id": getattr(el, "GlobalId", None),
                        "description": getattr(el, "Description", None),
                    }
                    
                    # Add property sets if requested
                    if include_psets:
                        try:
                            data["psets"] = ifcopenshell.util.element.get_psets(el)
                        except:
                            data["psets"] = {}
                    
                    building_elems.append(data)
            except:
                # Skip types not present in this model
                continue
        
        return building_elems
    
    def export_to_json(self, data: List[Dict], output_path: str, pretty: bool = True):
        """Export data to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        
        print(f"✓ Exported to JSON: {output_file}")
        return output_file
    
    def export_to_obj(self, output_path: str, elements: Optional[List] = None):
        """
        Export IFC geometry to OBJ format
        
        Args:
            output_path: Output OBJ file path
            elements: Specific elements to export (None = all products)
        """
        try:
            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_WORLD_COORDS, True)
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            vertices = []
            faces = []
            vertex_offset = 1  # OBJ indices start at 1
            
            # Get elements to process
            if elements is None:
                products = self.model.by_type("IfcProduct")
            else:
                products = elements
            
            count = 0
            for product in products:
                if not product.Representation:
                    continue
                
                try:
                    shape = ifcopenshell.geom.create_shape(settings, product)
                    verts = shape.geometry.verts
                    faces_raw = shape.geometry.faces
                    
                    # Add vertices (verts are flat: [x1,y1,z1, x2,y2,z2, ...])
                    for i in range(0, len(verts), 3):
                        vertices.append((verts[i], verts[i+1], verts[i+2]))
                    
                    # Add faces (faces are flat: [i1,i2,i3, i4,i5,i6, ...])
                    for i in range(0, len(faces_raw), 3):
                        faces.append((
                            faces_raw[i] + vertex_offset,
                            faces_raw[i+1] + vertex_offset,
                            faces_raw[i+2] + vertex_offset
                        ))
                    
                    vertex_offset += len(verts) // 3
                    count += 1
                    
                except Exception as e:
                    # Skip elements that can't be processed
                    continue
            
            # Write OBJ file
            with open(output_file, 'w') as f:
                f.write("# Exported from IFC Processor\n")
                f.write(f"# Source: {self.ifc_path.name}\n")
                f.write(f"# Elements: {count}\n\n")
                
                # Write vertices
                for v in vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
                # Write faces
                for face in faces:
                    f.write(f"f {face[0]} {face[1]} {face[2]}\n")
            
            print(f"✓ Exported to OBJ: {output_file} ({count} elements, {len(vertices)} vertices)")
            return output_file
            
        except Exception as e:
            print(f"✗ Error exporting to OBJ: {e}")
            return None
    
    def export_to_gltf(self, output_path: str):
        """
        Export to glTF format (requires additional setup)
        Note: This is a placeholder - full glTF export requires additional libraries
        """
        print("⚠ glTF export requires additional setup (e.g., Blender, trimesh)")
        print("  Recommended: Use Blender with BlenderBIM addon for glTF export")
        print(f"  Alternative: Export to OBJ first, then convert using Blender/online tools")
        return None
    
    def generate_simple_html_viewer(self, json_path: str, output_html: str = "viewer.html"):
        """
        Generate a simple HTML viewer for the JSON data
        This provides basic 2D/list view of elements
        """
        output_file = Path(output_html)
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IFC Element Viewer</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .stats {{
            background: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .search {{
            margin: 20px 0;
            padding: 10px;
            width: 100%;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #4CAF50;
            color: white;
            position: sticky;
            top: 0;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .type {{
            color: #1976D2;
            font-weight: bold;
        }}
        .id {{
            color: #666;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏗️ IFC Building Elements Viewer</h1>
        <div class="stats">
            <strong>Source:</strong> {self.ifc_path.name}<br>
            <strong>Total Elements:</strong> <span id="totalCount">0</span>
        </div>
        
        <input type="text" id="searchBox" class="search" placeholder="Search by type, name, or ID...">
        
        <table id="elementTable">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Type</th>
                    <th>Name</th>
                    <th>Global ID</th>
                </tr>
            </thead>
            <tbody id="tableBody">
            </tbody>
        </table>
    </div>
    
    <script>
        // Load JSON data
        fetch('{Path(json_path).name}')
            .then(response => response.json())
            .then(data => {{
                window.elementsData = data;
                renderTable(data);
                document.getElementById('totalCount').textContent = data.length;
            }})
            .catch(error => {{
                console.error('Error loading data:', error);
                document.getElementById('tableBody').innerHTML = 
                    '<tr><td colspan="4">Error loading data. Make sure the JSON file is in the same directory.</td></tr>';
            }});
        
        function renderTable(data) {{
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';
            
            data.forEach(elem => {{
                const row = tbody.insertRow();
                row.innerHTML = `
                    <td class="id">#${{elem.id}}</td>
                    <td class="type">${{elem.type}}</td>
                    <td>${{elem.name || '-'}}</td>
                    <td class="id">${{elem.global_id || '-'}}</td>
                `;
            }});
        }}
        
        // Search functionality
        document.getElementById('searchBox').addEventListener('input', function(e) {{
            const searchTerm = e.target.value.toLowerCase();
            const filtered = window.elementsData.filter(elem => 
                elem.type.toLowerCase().includes(searchTerm) ||
                (elem.name && elem.name.toLowerCase().includes(searchTerm)) ||
                elem.id.toString().includes(searchTerm) ||
                (elem.global_id && elem.global_id.toLowerCase().includes(searchTerm))
            );
            renderTable(filtered);
        }});
    </script>
</body>
</html>"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Generated HTML viewer: {output_file}")
        print(f"  Open in browser to view elements (requires JSON file in same directory)")
        return output_file
    
    def get_statistics(self, elements: List[Dict]) -> Dict[str, Any]:
        """Generate statistics about the elements"""
        stats = {
            "total_count": len(elements),
            "by_type": {},
            "with_names": 0,
            "with_psets": 0
        }
        
        for elem in elements:
            elem_type = elem["type"]
            stats["by_type"][elem_type] = stats["by_type"].get(elem_type, 0) + 1
            
            if elem.get("name"):
                stats["with_names"] += 1
            
            if elem.get("psets"):
                stats["with_psets"] += 1
        
        return stats


def main():
    """Main execution function"""
    
    # Configuration
    ifc_path = "sample/Architectural.ifc"  # Change to your file
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("IFC PROCESSOR - Research Prototype")
    print("=" * 60)
    print()
    
    try:
        # Initialize processor
        processor = IFCProcessor(ifc_path)
        
        # 1. Extract building elements (real elements only)
        print("\n[1/5] Extracting building elements...")
        building_elements = processor.extract_building_elements(include_psets=True)
        print(f"Found {len(building_elements)} building elements")
        
        # Show statistics
        stats = processor.get_statistics(building_elements)
        print("\nElement breakdown:")
        for elem_type, count in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
            print(f"  {elem_type}: {count}")
        
        # 2. Export to JSON
        print("\n[2/5] Exporting to JSON...")
        json_file = processor.export_to_json(
            building_elements,
            output_dir / "building_elements.json"
        )
        
        # Also export first 10 for preview
        processor.export_to_json(
            building_elements[:10],
            output_dir / "building_elements_preview.json"
        )
        print(f"  Preview (first 10): {output_dir / 'building_elements_preview.json'}")
        
        # 3. Export to OBJ (3D visualization format)
        print("\n[3/5] Exporting to OBJ (3D format)...")
        obj_file = processor.export_to_obj(
            output_dir / "building_model.obj"
        )
        
        # 4. Generate HTML viewer (2D/list view)
        print("\n[4/5] Generating HTML viewer...")
        html_file = processor.generate_simple_html_viewer(
            json_file,
            output_dir / "element_viewer.html"
        )
        
        # 5. Summary
        print("\n[5/5] Summary")
        print("=" * 60)
        print(f"✓ Total building elements extracted: {len(building_elements)}")
        print(f"✓ Element types found: {len(stats['by_type'])}")
        print(f"✓ Elements with names: {stats['with_names']}")
        print(f"✓ Elements with property sets: {stats['with_psets']}")
        print()
        print("Output files:")
        print(f"  📄 JSON data: {output_dir / 'building_elements.json'}")
        print(f"  📄 JSON preview: {output_dir / 'building_elements_preview.json'}")
        if obj_file:
            print(f"  🔷 3D model (OBJ): {obj_file}")
        print(f"  🌐 HTML viewer: {html_file}")
        print()
        print("Next steps:")
        print("  • View JSON data for schema mapping (T2)")
        print("  • Open OBJ in Blender/MeshLab for 3D visualization")
        print("  • Open HTML viewer in browser for element browsing")
        print("  • Use GlobalId for photo→IFC linking (T1)")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"\nPlease ensure your IFC file exists at: {ifc_path}")
        print("Or update the 'ifc_path' variable in the script")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()