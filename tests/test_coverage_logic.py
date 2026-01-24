"""Comprehensive coverage for the coverage check utility."""
import xml.etree.ElementTree as ET
from unittest import mock
import check_coverage


def test_run_report_success():
    """Test run_report with a valid coverage.xml."""
    xml_content = """<coverage line-rate="0.95">
        <packages>
            <package name="modules">
                <classes>
                    <class filename="modules/mod1.py" line-rate="0.9" name="mod1.py"></class>
                </classes>
            </package>
        </packages>
    </coverage>"""
    root = ET.fromstring(xml_content)

    with mock.patch("xml.etree.ElementTree.parse") as mock_parse:
        mock_tree = mock.MagicMock()
        mock_tree.getroot.return_value = root
        mock_parse.return_value = mock_tree

        with mock.patch("check_coverage.logger") as mock_logger:
            check_coverage.run_report()
            # Just check that it was called with the right data
            mock_logger.info.assert_any_call(
                mock.ANY, mock.ANY, "modules/mod1.py", 90.0)
            mock_logger.info.assert_any_call(mock.ANY, 95.0)


def test_run_report_exception():
    """Test run_report handles exceptions."""
    with mock.patch("xml.etree.ElementTree.parse", side_effect=Exception("parse error")):
        with mock.patch("check_coverage.logger") as mock_logger:
            check_coverage.run_report()
            mock_logger.error.assert_called()
