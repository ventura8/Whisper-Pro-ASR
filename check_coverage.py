"""Check coverage from XML file."""
import xml.etree.ElementTree as ET
import logging
import sys

# Configure logging for simple report output
logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def run_report():
    try:
        tree = ET.parse('coverage.xml')
        root = tree.getroot()

        logger.info("=" * 60)
        logger.info("COVERAGE REPORT BY FILE")
        logger.info("=" * 60)

        packages = root.find('packages')
        files = []
        for p in packages:
            for c in p.findall('.//class'):
                filename = c.get('filename')
                rate = float(c.get('line-rate')) * 100
                files.append((filename, rate))

        files.sort(key=lambda x: x[1])

        for filename, rate in files:
            status = "✓" if rate >= 90 else "✗"
            logger.info("%s %s: %.0f%%", status, filename, rate)

        logger.info("=" * 60)
        total_rate = float(root.get('line-rate')) * 100
        logger.info("TOTAL: %.0f%%", total_rate)
    except Exception as e:
        logger.error("Error parsing coverage: %s", e)

if __name__ == "__main__":
    run_report()
