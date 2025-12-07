import subprocess
import os

def run_blender_render(ifc_path, target_guid):
    """
    启动 Headless Blender 子进程来渲染指定 GUID。
    """
    # 1. 路径配置 (请修改为你的真实路径)
    # BLENDER_EXE = "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe" # Windows 示例
    # BLENDER_EXE = "/Applications/Blender.app/Contents/MacOS/Blender" # Mac 示例
    BLENDER_EXR = "/mnt/d/Program Files/Blender Foundation/Blender 5.0/blender.exe"
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    SCRIPT_PATH = os.path.join(PROJECT_ROOT, "blender_scripts", "render_worker.py")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "renders")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    output_filename = os.path.join(OUTPUT_DIR, f"{target_guid}.png")

    # 2. 构建命令
    # blender -b (后台) -P (运行脚本) -- (传递参数)
    cmd = [
        BLENDER_EXE,
        "-b",
        "-P", SCRIPT_PATH,
        "--",
        ifc_path,
        output_filename,
        target_guid
    ]

    print(f"[Blender Service] Rendering {target_guid}...")
    
    # 3. 执行并等待
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # print(result.stdout) # 调试时打开
        return output_filename
    except subprocess.CalledProcessError as e:
        print(f"Blender Error: {e.stderr}")
        return "Error: Rendering failed."