import subprocess
from typing import Annotated, List
from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP()

def run_applescript(script):
    p = subprocess.Popen(["osascript", "-e", script],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    return stdout.decode("utf-8").strip(), stderr.decode("utf-8").strip()


@mcp.tool(name="close_terminal_if_open", description="close Terminal.app if it is currently running")
def close_terminal_if_open() -> str:
    stdout, stderr = run_applescript("""
    tell application "System Events"
        if exists process "Terminal" then
            tell application "Terminal" to quit
        end if
    end tell
    """)
    if stderr:
        return f"AppleScript err: {stderr}"
    return "Terminal closed successfully"


@mcp.tool(name="open_new_terminal", description="open or activate Terminal.app, optionally focusing a window by id")
def open_new_terminal(window_id: Annotated[str, Field(description="optional Terminal window id to focus", examples="1")] = "") -> str:
    if window_id:
        stdout, stderr = run_applescript(f"""
tell application "Terminal"
    if (count of windows) > 0 then
        set theWindow to window id {window_id}
        set frontmost of theWindow to true
        activate
    else
        activate
    end if
end tell
""")
    else:
        stdout, stderr = run_applescript("""
tell application "Terminal"
    activate
end tell
""")
    if stderr:
        return f"AppleScript err: {stderr}"
    return get_all_terminal_window_ids()


@mcp.tool(name="get_all_terminal_window_ids", description="list all Terminal windows and their tabs")
def get_all_terminal_window_ids() -> str:
    stdout, stderr = run_applescript("""
    tell application "Terminal"
        set outputList to {}
        repeat with aWindow in windows
            set windowID to id of aWindow
            set tabCount to number of tabs of aWindow
            repeat with tabIndex from 1 to tabCount
                set end of outputList to {tab tabIndex of window id windowID}
            end repeat
        end repeat
    end tell
    return outputList
""")
    if stderr:
        return f"AppleScript err: {stderr}"
    return stdout


@mcp.tool(name="run_script_in_terminal", description="run a shell script in the front Terminal window")
def run_script_in_terminal(script: Annotated[str, Field(description="shell script to run in Terminal", examples="pwd")]) -> str:
    stdout, stderr = run_applescript(f"""
tell application "Terminal"
    activate
    if (count of windows) > 0 then
        do script "{script}" in window 1
    else
        do script "{script}"
    end if
end tell
""")
    if stderr:
        return f"AppleScript err: {stderr}"
    return stdout


@mcp.tool(name="get_terminal_full_text", description="get the full text history of the selected tab in the front Terminal window")
def get_terminal_full_text() -> str:
    stdout, stderr = run_applescript("""
tell application "Terminal"
    set fullText to history of selected tab of front window
end tell
""")
    if stderr:
        return f"AppleScript err: {stderr}"
    return stdout

def parse_key_code(button):
    button = button.lower()

    keycode_map = {
        'return': 'return',
        'space': 'space',
        'tab': 'tab',
        'delete': 'delete',
        'escape': 'escape',
        'up': 126,
        'down': 125,
        'left': 123,
        'right': 124,
        'a': 0,
        'b': 11,
        'c': 8,
        'd': 2,
        'e': 14,
        'f': 3,
        'g': 5,
        'h': 4,
        'i': 34,
        'j': 38,
        'k': 40,
        'l': 37,
        'm': 46,
        'n': 45,
        'o': 31,
        'p': 35,
        'q': 12,
        'r': 15,
        's': 1,
        't': 17,
        'u': 32,
        'v': 9,
        'w': 13,
        'x': 7,
        'y': 16,
        'z': 6,
        '0': 29,
        '1': 18,
        '2': 19,
        '3': 20,
        '4': 21,
        '5': 23,
        '6': 22,
        '7': 26,
        '8': 28,
        '9': 25,
    }

    return keycode_map.get(button, button)


def concat_key_codes(key_codes):
    script = ''
    for key in key_codes:
        key_code = parse_key_code(key)
        if isinstance(key_code, int):
            script += f'key code {key_code}\n'
        else:
            script += f'keystroke {key_code}\n'
        script += 'delay 0.5\n'
    return script.strip()


@mcp.tool(name="send_terminal_keyboard_key", description="send a terminal keyboard key to an existing terminal")
def send_terminal_keyboard_key(key_codes: Annotated[List[str], Field(description="sequence of keys to send, e.g. letters, digits, 'return', 'space', 'tab', 'delete', 'escape', 'up', 'down', 'left', 'right'", examples=[["h", "i", "return"]])]) -> bool:
    print('\nsend_terminal_keyboard_key keycode:', key_codes)
    print('-' * 50)
    script = f'''
tell application "Terminal"
    activate
    tell application "System Events"
        {concat_key_codes(key_codes)}
    end tell
end tell
'''
    print(script)
    terminal_content, error = run_applescript(script)
    if error:
        return False
    else:
        return True


if __name__ == "__main__":
    # send_terminal_keyboard_key(["return"])
    mcp.run(transport="stdio")

    # close_terminal_if_open()
    # window_ids = open_new_terminal()
    # print(window_ids)
    # run_script_in_terminal("pwd")

    # full_text = get_terminal_full_text()
    # print(full_text)
