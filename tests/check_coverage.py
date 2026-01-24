"""Enforce per-file coverage threshold from coverage.xml."""
import xml.etree.ElementTree as ET
import sys


def check_coverage(xml_file, threshold=0.9, use_color=True):
    """Check if all files in the coverage.xml meet the threshold."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    failed = False

    print("\n" + "="*60)
    print(f"{'FILE':<40} | {'COVERAGE':<10} | {'STATUS':<6}")
    print("-" * 60)

    # Style
    clr = {
        'green': "\033[92m" if use_color else "",
        'red': "\033[91m" if use_color else "",
        'reset': "\033[0m" if use_color else "",
        'bold': "\033[1m" if use_color else ""
    }

    for package in root.findall('.//package'):
        for cls in package.findall('.//class'):
            line_rate = float(cls.get('line-rate'))
            filename = cls.get('filename')
            pct = line_rate * 100

            if line_rate < threshold:
                status = f"{clr['red']}FAIL{clr['reset']}"
                failed = True
            else:
                status = f"{clr['green']}PASS{clr['reset']}"

            print(f"{filename:<40} | {pct:>8.2f}% | {status}")

    print("-" * 60)
    if failed:
        msg = f"CRITICAL: One or more files are below the {threshold*100:.0f}% threshold!"
        print(f"{clr['bold']}{clr['red']}{msg}{clr['reset']}")
        sys.exit(1)
    else:
        msg = f"SUCCESS: All files exceed the {threshold*100:.0f}% threshold."
        print(f"{clr['bold']}{clr['green']}{msg}{clr['reset']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Check if --no-color is passed
    no_color = "--no-color" in sys.argv
    check_coverage('coverage.xml', use_color=not no_color)
