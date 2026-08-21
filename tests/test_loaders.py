from unittest.mock import MagicMock, patch

from loaders import load_documents


class TestLoadTxt:
    def test_loads_txt_file(self, tmp_path):
        (tmp_path / "hello.txt").write_text("Hello world")
        results = load_documents(tmp_path)
        assert len(results) == 1
        assert results[0]["filename"] == "hello.txt"
        assert results[0]["text"] == "Hello world"

    def test_loads_md_file(self, tmp_path):
        (tmp_path / "readme.md").write_text("# Title\nSome content")
        results = load_documents(tmp_path)
        assert len(results) == 1
        assert results[0]["filename"] == "readme.md"
        assert "# Title" in results[0]["text"]


class TestLoadPdf:
    def test_loads_pdf_with_text(self, tmp_path):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF content here"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("loaders.PdfReader", return_value=mock_reader):
            (tmp_path / "test.pdf").write_bytes(b"%PDF-1.4 fake")
            results = load_documents(tmp_path)

        assert len(results) == 1
        assert results[0]["filename"] == "test.pdf"
        assert results[0]["text"] == "PDF content here"

    def test_loads_pdf_with_no_text(self, tmp_path):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("loaders.PdfReader", return_value=mock_reader):
            (tmp_path / "empty.pdf").write_bytes(b"%PDF-1.4 fake")
            results = load_documents(tmp_path)

        assert len(results) == 1
        assert results[0]["filename"] == "empty.pdf"
        assert results[0]["text"] == ""

    def test_loads_multi_page_pdf(self, tmp_path):
        pages = []
        for i in range(3):
            page = MagicMock()
            page.extract_text.return_value = f"Page {i} content"
            pages.append(page)
        mock_reader = MagicMock()
        mock_reader.pages = pages

        with patch("loaders.PdfReader", return_value=mock_reader):
            (tmp_path / "multipage.pdf").write_bytes(b"%PDF-1.4 fake")
            results = load_documents(tmp_path)

        assert len(results) == 1
        assert "Page 0 content" in results[0]["text"]
        assert "Page 2 content" in results[0]["text"]

    def test_loads_pdf_with_mixed_pages(self, tmp_path):
        page_with_text = MagicMock()
        page_with_text.extract_text.return_value = "Page with text"
        page_empty = MagicMock()
        page_empty.extract_text.return_value = ""
        page_with_text2 = MagicMock()
        page_with_text2.extract_text.return_value = "Another page"

        mock_reader = MagicMock()
        mock_reader.pages = [page_with_text, page_empty, page_with_text2]

        with patch("loaders.PdfReader", return_value=mock_reader):
            (tmp_path / "mixed.pdf").write_bytes(b"%PDF-1.4 fake")
            results = load_documents(tmp_path)

        assert len(results) == 1
        assert "Page with text" in results[0]["text"]
        assert "Another page" in results[0]["text"]
        assert results[0]["text"].count("\n") == 1  # joined by \n


class TestLoadEdgeCases:
    def test_empty_directory(self, tmp_path):
        results = load_documents(tmp_path)
        assert results == []

    def test_skips_unsupported_extension(self, tmp_path):
        (tmp_path / "data.csv").write_text("col1,col2")
        (tmp_path / "readme.txt").write_text("Supported")
        results = load_documents(tmp_path)
        assert len(results) == 1
        assert results[0]["filename"] == "readme.txt"

    def test_multiple_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("File A")
        (tmp_path / "b.md").write_text("File B")
        (tmp_path / "c.txt").write_text("File C")
        results = load_documents(tmp_path)
        filenames = {r["filename"] for r in results}
        assert filenames == {"a.txt", "b.md", "c.txt"}
