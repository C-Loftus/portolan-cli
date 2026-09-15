"""Tests for ImageServer extraction orchestrator.

Tests verify the full extraction pipeline using Wave 1 data models.
Uses mocking for HTTP and COG conversion to keep tests fast and isolated.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from portolan_cli.extract.arcgis.imageserver.discovery import ImageServerMetadata
from portolan_cli.extract.arcgis.imageserver.extractor import (
    ExtractionConfig,
    ExtractionResult,
    ImageServerExtractionError,
    download_tile,
    extract_imageserver,
)
from portolan_cli.extract.arcgis.imageserver.tiling import TileSpec

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_metadata() -> ImageServerMetadata:
    """Standard ImageServer metadata for testing."""
    return ImageServerMetadata(
        name="TestImageServer",
        band_count=1,
        pixel_type="F32",
        pixel_size_x=10.0,
        pixel_size_y=10.0,
        full_extent={
            "xmin": 0,
            "ymin": 0,
            "xmax": 10000,
            "ymax": 10000,
            "spatialReference": {"wkid": 4326},
        },
        max_image_width=4096,
        max_image_height=4096,
        capabilities=["Image", "Metadata"],
        description="Test service",
    )


@pytest.fixture
def small_extent_metadata() -> ImageServerMetadata:
    """Metadata with small extent (single tile)."""
    return ImageServerMetadata(
        name="SmallService",
        band_count=1,
        pixel_type="U8",
        pixel_size_x=1.0,
        pixel_size_y=1.0,
        full_extent={
            "xmin": 0,
            "ymin": 0,
            "xmax": 100,
            "ymax": 100,
            "spatialReference": {"wkid": 4326},
        },
        max_image_width=4096,
        max_image_height=4096,
        capabilities=["Image"],
    )


@pytest.fixture
def sample_tile() -> TileSpec:
    """Sample tile for download tests."""
    return TileSpec(
        x=0,
        y=0,
        bbox=(0.0, 0.0, 4096.0, 4096.0),
        width_px=4096,
        height_px=4096,
    )


# =============================================================================
# ExtractionConfig Tests
# =============================================================================


@pytest.mark.unit
class TestExtractionConfig:
    """Tests for ExtractionConfig dataclass."""

    def test_default_tile_size(self) -> None:
        """Default tile size is 4096."""
        config = ExtractionConfig()
        assert config.tile_size == 4096

    def test_default_compression(self) -> None:
        """Default compression is DEFLATE (via cog_settings)."""
        config = ExtractionConfig()
        # compression is now in cog_settings
        assert config.cog_settings.compression == "DEFLATE"
        # Legacy field is None when using cog_settings
        assert config.compression is None

    def test_default_max_retries(self) -> None:
        """Default max retries is 3."""
        config = ExtractionConfig()
        assert config.max_retries == 3

    def test_default_dry_run_false(self) -> None:
        """Default dry_run is False."""
        config = ExtractionConfig()
        assert config.dry_run is False

    def test_default_raw_false(self) -> None:
        """Default raw is False (auto-init catalog by default)."""
        config = ExtractionConfig()
        assert config.raw is False

    def test_custom_values(self) -> None:
        """Custom config values are preserved."""
        from portolan_cli.conversion_config import CogSettings

        config = ExtractionConfig(
            tile_size=2048,
            cog_settings=CogSettings(compression="JPEG", quality=85),
            max_retries=5,
            dry_run=True,
        )
        assert config.tile_size == 2048
        assert config.cog_settings.compression == "JPEG"
        assert config.cog_settings.quality == 85
        assert config.max_retries == 5
        assert config.dry_run is True


# =============================================================================
# ExtractionResult Tests
# =============================================================================


@pytest.mark.unit
class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_result_attributes(self) -> None:
        """ExtractionResult has expected attributes."""
        result = ExtractionResult(
            output_dir=Path("/output"),
            tiles_downloaded=10,
            tiles_skipped=0,
            total_bytes=1024000,
            catalog_initialized=True,
        )
        assert result.output_dir == Path("/output")
        assert result.tiles_downloaded == 10
        assert result.tiles_skipped == 0
        assert result.total_bytes == 1024000
        assert result.catalog_initialized is True

    def test_result_defaults(self) -> None:
        """ExtractionResult has sensible defaults."""
        result = ExtractionResult(
            output_dir=Path("/output"),
            tiles_downloaded=5,
            tiles_skipped=0,
        )
        assert result.tiles_failed == 0
        assert result.total_bytes == 0
        assert result.catalog_initialized is False


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.unit
class TestErrorHandling:
    """Tests for error handling in extraction."""

    def test_extraction_error_is_exception(self) -> None:
        """ImageServerExtractionError is an Exception."""
        error = ImageServerExtractionError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_extraction_error_with_cause(self) -> None:
        """ImageServerExtractionError can wrap another exception."""
        cause = ValueError("Original error")
        error = ImageServerExtractionError("Wrapped error")
        error.__cause__ = cause
        assert error.__cause__ is cause


# =============================================================================
# Async Function Tests (proper pytest-asyncio)
# =============================================================================


@pytest.mark.unit
class TestDownloadTile:
    """Tests for download_tile async function."""

    # Valid TIFF header (little-endian) for mock responses
    # Magic bytes II (0x4949) + version 42 (0x002A) + offset to first IFD
    VALID_TIFF_HEADER = b"II\x2a\x00" + b"\x08\x00\x00\x00" + b"\x00" * 100

    @pytest.mark.asyncio
    async def test_download_builds_correct_url(self, sample_tile: TileSpec, tmp_path: Path) -> None:
        """Download constructs correct exportImage URL."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        # Use valid TIFF header to pass validation
        mock_response.content = self.VALID_TIFF_HEADER
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response

        output_path = tmp_path / "out.tif"
        url = "https://example.com/ImageServer"
        await download_tile(url, sample_tile, output_path, mock_client)

        # Verify URL was called
        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        called_url = call_args[0][0] if call_args[0] else str(call_args)
        # URL should contain exportImage
        assert "exportImage" in called_url

    @pytest.mark.asyncio
    async def test_download_returns_bytes_count(
        self, sample_tile: TileSpec, tmp_path: Path
    ) -> None:
        """Download returns number of bytes downloaded."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        # Use valid TIFF header + padding to get 1000 bytes
        content = self.VALID_TIFF_HEADER + b"x" * (1000 - len(self.VALID_TIFF_HEADER))
        mock_response.content = content
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response

        output_path = tmp_path / "out.tif"
        result = await download_tile(
            "https://example.com/ImageServer",
            sample_tile,
            output_path,
            mock_client,
        )

        assert result == 1000
        # Verify file was actually written
        assert output_path.exists()
        assert output_path.read_bytes() == content


@pytest.mark.unit
class TestExtractImageserver:
    """Tests for extract_imageserver async function."""

    @pytest.mark.asyncio
    async def test_dry_run_returns_result(
        self, tmp_path: Path, small_extent_metadata: ImageServerMetadata
    ) -> None:
        """Dry run returns ExtractionResult without downloads."""
        with patch(
            "portolan_cli.extract.arcgis.imageserver.extractor.discover_imageserver",
            new_callable=AsyncMock,
        ) as mock_discover:
            mock_discover.return_value = small_extent_metadata

            config = ExtractionConfig(dry_run=True)
            result = await extract_imageserver(
                "https://example.com/ImageServer",
                tmp_path,
                config=config,
            )

        assert isinstance(result, ExtractionResult)
        assert result.tiles_downloaded == 0

    @pytest.mark.asyncio
    async def test_extraction_with_bbox_filter(
        self, tmp_path: Path, sample_metadata: ImageServerMetadata
    ) -> None:
        """Extraction accepts bbox filter parameter."""
        with patch(
            "portolan_cli.extract.arcgis.imageserver.extractor.discover_imageserver",
            new_callable=AsyncMock,
        ) as mock_discover:
            mock_discover.return_value = sample_metadata

            config = ExtractionConfig(dry_run=True)
            result = await extract_imageserver(
                "https://example.com/ImageServer",
                tmp_path,
                config=config,
                bbox=(0, 0, 100, 100),
            )

        assert isinstance(result, ExtractionResult)

    @pytest.mark.asyncio
    async def test_discovery_error_propagates(self, tmp_path: Path) -> None:
        """Discovery errors propagate correctly."""
        from portolan_cli.extract.arcgis.imageserver.discovery import (
            ImageServerDiscoveryError,
        )

        with patch(
            "portolan_cli.extract.arcgis.imageserver.extractor.discover_imageserver",
            new_callable=AsyncMock,
        ) as mock_discover:
            mock_discover.side_effect = ImageServerDiscoveryError("Connection failed")

            with pytest.raises((ImageServerExtractionError, ImageServerDiscoveryError)):
                await extract_imageserver(
                    "https://invalid.example.com/ImageServer",
                    tmp_path,
                )


# =============================================================================
# Integration-style Tests (still unit, but test module interactions)
# =============================================================================


@pytest.mark.unit
class TestModuleImports:
    """Tests verifying module structure and imports."""

    def test_all_exports_importable(self) -> None:
        """All __all__ exports are importable."""
        from portolan_cli.extract.arcgis.imageserver import (
            ExtractionConfig,
            ExtractionResult,
            ImageServerExtractionError,
            download_tile,
            extract_imageserver,
        )

        # Just verify they're the right types
        assert ExtractionConfig is not None
        assert ExtractionResult is not None
        assert ImageServerExtractionError is not None
        assert callable(download_tile)
        assert callable(extract_imageserver)

    def test_extraction_config_is_dataclass(self) -> None:
        """ExtractionConfig is a proper dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(ExtractionConfig)

    def test_extraction_result_is_dataclass(self) -> None:
        """ExtractionResult is a proper dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(ExtractionResult)


# =============================================================================
# Tests for Issue #335 Fixes
# =============================================================================


@pytest.mark.unit
class TestBboxCrsDetection:
    """Tests for WGS84 bbox detection and reprojection (issue #335 fix)."""

    def test_is_likely_wgs84_with_lat_lon_coords(self) -> None:
        """Bbox with WGS84-range coordinates is detected."""
        from portolan_cli.extract.arcgis.imageserver.extractor import _is_likely_wgs84

        # Philadelphia area in WGS84
        bbox = (-75.17, 39.95, -75.15, 39.97)
        assert _is_likely_wgs84(bbox) is True

    def test_is_likely_wgs84_with_web_mercator_coords(self) -> None:
        """Bbox with Web Mercator coordinates is NOT detected as WGS84."""
        from portolan_cli.extract.arcgis.imageserver.extractor import _is_likely_wgs84

        # Philadelphia area in Web Mercator (large numbers)
        bbox = (-8367886, 4858679, -8365659, 4861583)
        assert _is_likely_wgs84(bbox) is False

    def test_is_likely_wgs84_edge_case_poles(self) -> None:
        """Bbox at edge of WGS84 range is detected."""
        from portolan_cli.extract.arcgis.imageserver.extractor import _is_likely_wgs84

        # Global extent
        bbox = (-180, -90, 180, 90)
        assert _is_likely_wgs84(bbox) is True

    def test_reproject_bbox_wgs84_to_web_mercator(self) -> None:
        """Bbox is correctly reprojected from WGS84 to Web Mercator."""
        from portolan_cli.extract.arcgis.imageserver.extractor import _reproject_bbox

        # Philadelphia area: known coordinates for verification
        # WGS84: (-75.17, 39.95, -75.15, 39.97)
        # Expected Web Mercator (approximately):
        # minx: -8367886, miny: 4858679, maxx: -8365659, maxy: 4861583
        bbox = (-75.17, 39.95, -75.15, 39.97)
        result = _reproject_bbox(bbox, "EPSG:4326", "EPSG:3857")

        # Verify against known correct values (within 100m tolerance)
        assert -8368000 < result[0] < -8367000  # minx ~ -8367886
        assert 4858000 < result[1] < 4859000  # miny ~ 4858679
        assert -8366000 < result[2] < -8365000  # maxx ~ -8365659
        assert 4861000 < result[3] < 4862000  # maxy ~ 4861583

    def test_reproject_bbox_if_needed_passthrough_for_wgs84_service(self) -> None:
        """Bbox is not reprojected if service is already WGS84."""
        from portolan_cli.extract.arcgis.imageserver.extractor import reproject_bbox_if_needed

        bbox = (-75.17, 39.95, -75.15, 39.97)
        result = reproject_bbox_if_needed(bbox, "EPSG:4326")

        # Should be unchanged
        assert result == bbox

    def test_reproject_bbox_if_needed_converts_wgs84_to_service_crs(self) -> None:
        """WGS84 bbox is auto-reprojected to service CRS."""
        from portolan_cli.extract.arcgis.imageserver.extractor import reproject_bbox_if_needed

        # WGS84 coords (Philadelphia)
        bbox = (-75.17, 39.95, -75.15, 39.97)
        result = reproject_bbox_if_needed(bbox, "EPSG:3857")

        # Verify against known correct Web Mercator values
        assert -8368000 < result[0] < -8367000  # minx ~ -8367886
        assert 4858000 < result[1] < 4859000  # miny ~ 4858679
        assert -8366000 < result[2] < -8365000  # maxx ~ -8365659
        assert 4861000 < result[3] < 4862000  # maxy ~ 4861583

    def test_reproject_bbox_if_needed_explicit_bbox_crs(self) -> None:
        """Explicit bbox_crs parameter overrides auto-detection."""
        from portolan_cli.extract.arcgis.imageserver.extractor import reproject_bbox_if_needed

        # State Plane coords that happen to be in WGS84 range (would trigger false positive)
        bbox = (100.0, 50.0, 150.0, 80.0)

        # Without explicit bbox_crs, this would be detected as WGS84 and reprojected
        # With explicit bbox_crs matching service CRS, no reprojection happens
        result = reproject_bbox_if_needed(bbox, "EPSG:3857", bbox_crs="EPSG:3857")

        # Should be unchanged (same CRS)
        assert result == bbox

    def test_reproject_bbox_if_needed_explicit_bbox_crs_different(self) -> None:
        """Explicit bbox_crs triggers reprojection when different from service CRS."""
        from portolan_cli.extract.arcgis.imageserver.extractor import reproject_bbox_if_needed

        # Explicit WGS84 bbox
        bbox = (-75.17, 39.95, -75.15, 39.97)
        result = reproject_bbox_if_needed(bbox, "EPSG:3857", bbox_crs="EPSG:4326")

        # Should be reprojected to Web Mercator
        assert -8368000 < result[0] < -8367000


@pytest.mark.unit
class TestCollectionNameValidation:
    """Tests for collection name validation (path traversal prevention)."""

    def test_validate_collection_name_simple(self) -> None:
        """Simple collection names pass validation."""
        from portolan_cli.extract.arcgis.imageserver.extractor import _validate_collection_name

        assert _validate_collection_name("tiles") == "tiles"
        assert _validate_collection_name("naip-2024") == "naip-2024"
        assert _validate_collection_name("my_collection") == "my_collection"

    def test_validate_collection_name_strips_path_components(self) -> None:
        """Path traversal attempts are sanitized."""
        from portolan_cli.extract.arcgis.imageserver.extractor import _validate_collection_name

        # Path traversal attempts get stripped to just the base name
        assert _validate_collection_name("../../../etc") == "etc"
        assert _validate_collection_name("/etc/passwd") == "passwd"
        assert _validate_collection_name("foo/bar/baz") == "baz"

    def test_validate_collection_name_rejects_empty(self) -> None:
        """Empty names are rejected."""
        from portolan_cli.extract.arcgis.imageserver.extractor import _validate_collection_name

        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_collection_name("")

        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_collection_name(".")

        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_collection_name("..")

    def test_validate_collection_name_rejects_invalid_chars(self) -> None:
        """Names with invalid characters are rejected."""
        from portolan_cli.extract.arcgis.imageserver.extractor import _validate_collection_name

        with pytest.raises(ValueError, match="cannot contain"):
            _validate_collection_name("foo<bar")

        with pytest.raises(ValueError, match="cannot contain"):
            _validate_collection_name("foo|bar")

        with pytest.raises(ValueError, match="cannot contain"):
            _validate_collection_name("foo?bar")


@pytest.mark.unit
class TestJsonErrorParsing:
    """Tests for JSON error response parsing (issue #335 fix)."""

    @pytest.mark.asyncio
    async def test_download_tile_parses_arcgis_json_error(
        self, sample_tile: TileSpec, tmp_path: Path
    ) -> None:
        """JSON error responses from ArcGIS are parsed correctly."""
        from portolan_cli.extract.arcgis.imageserver.extractor import (
            ImageServerExtractionError,
            download_tile,
        )

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        # Simulate ArcGIS JSON error response (not a TIFF)
        error_json = (
            b'{"error":{"code":400,"message":"The requested image exceeds the size limit."}}'
        )
        mock_response.content = error_json
        mock_response.status_code = 200  # ArcGIS returns 200 with error in body
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response

        output_path = tmp_path / "out.tif"
        with pytest.raises(ImageServerExtractionError) as exc_info:
            await download_tile(
                "https://example.com/ImageServer",
                sample_tile,
                output_path,
                mock_client,
            )

        # Error message should contain the ArcGIS error details
        assert "400" in str(exc_info.value)
        assert "exceeds the size limit" in str(exc_info.value)


@pytest.mark.unit
class TestAutoInitCatalogExistingCatalog:
    """Tests for _auto_init_catalog on an existing catalog (issue #767, #832)."""

    def _make_tile(self, output_dir: Path, collection_name: str = "tiles") -> None:
        """Write a placeholder COG so _auto_init_catalog finds a file to add."""
        item_dir = output_dir / collection_name / "tile_0_0"
        item_dir.mkdir(parents=True)
        (item_dir / "tile_0_0.tif").write_bytes(b"fake-cog")

    def test_first_extraction_initializes_catalog(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fresh directory calls init_catalog, then add_files."""
        from portolan_cli.extract.arcgis.imageserver.extractor import _auto_init_catalog

        self._make_tile(tmp_path)
        events: list[str] = []

        import portolan_cli.add as add_mod
        import portolan_cli.catalog as catalog_mod

        def _fake_init(output_dir: Path, **kwargs: object) -> tuple[Path, list[str]]:
            events.append("init")
            return output_dir / "catalog.json", []

        monkeypatch.setattr(catalog_mod, "init_catalog", _fake_init)
        monkeypatch.setattr(add_mod, "add_files", lambda **k: events.append("add"))

        result = _auto_init_catalog(tmp_path, service_name="svc")

        assert result is True
        assert events == ["init", "add"]

    def test_second_extraction_skips_init_on_existing_catalog(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second ImageServer extraction adds tiles without aborting (issue #767).

        The COG tiles download before _auto_init_catalog runs. Before the fix,
        init_catalog raised CatalogAlreadyExistsError on a MANAGED directory, so
        the run aborted after the download and never wrote collection.json. The
        guard must skip init_catalog and add the tiles instead.
        """
        from portolan_cli.extract.arcgis.imageserver.extractor import _auto_init_catalog

        # Mark the directory as an existing Portolan catalog. detect_state reads
        # config.yaml alone to return MANAGED.
        (tmp_path / ".portolan").mkdir()
        (tmp_path / ".portolan" / "config.yaml").write_text("# Portolan configuration\n")

        self._make_tile(tmp_path, collection_name="naip-2024")
        events: list[str] = []

        import portolan_cli.add as add_mod
        import portolan_cli.catalog as catalog_mod

        monkeypatch.setattr(
            catalog_mod, "init_catalog", lambda *a, **k: (tmp_path / "catalog.json", [])
        )
        monkeypatch.setattr(add_mod, "add_files", lambda **k: events.append("add"))

        result = _auto_init_catalog(tmp_path, service_name="svc", collection_name="naip-2024")

        assert result is True
        # init_catalog is skipped for an existing catalog; the tiles are added.
        assert events == ["add"]

    def test_reports_init_catalog_warnings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The raster path prints what init_catalog had to guess (issue #821)."""
        from portolan_cli.extract.arcgis.imageserver.extractor import _auto_init_catalog

        self._make_tile(tmp_path)

        import portolan_cli.add as add_mod
        import portolan_cli.catalog as catalog_mod

        def _fake_init(output_dir: Path, **kwargs: object) -> tuple[Path, list[str]]:
            return output_dir / "catalog.json", ["Derived catalog id 'publish'"]

        monkeypatch.setattr(catalog_mod, "init_catalog", _fake_init)
        monkeypatch.setattr(add_mod, "add_files", lambda **k: None)

        _auto_init_catalog(tmp_path, service_name="svc")

        assert "Derived catalog id 'publish'" in capsys.readouterr().err

    def test_warns_when_id_cannot_apply_to_an_existing_catalog(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An existing catalog keeps its id, so a passed --id must not go silent."""
        from portolan_cli.extract.arcgis.imageserver.extractor import _auto_init_catalog

        (tmp_path / ".portolan").mkdir()
        (tmp_path / ".portolan" / "config.yaml").write_text("# Portolan configuration\n")
        self._make_tile(tmp_path)

        import portolan_cli.add as add_mod

        monkeypatch.setattr(add_mod, "add_files", lambda **k: None)

        _auto_init_catalog(tmp_path, service_name="svc", catalog_id="phl-housing")

        assert "Ignored --id 'phl-housing'" in capsys.readouterr().err


# =============================================================================
# HTTP error messages (issue #870)
# =============================================================================


def _mock_client_with_status(status_code: int, content: bytes) -> AsyncMock:
    """Build an httpx client mock whose GET fails raise_for_status with a body."""
    request = httpx.Request("GET", "https://example.com/ImageServer/exportImage")
    response = httpx.Response(status_code, content=content, request=request)
    mock_response = AsyncMock()
    mock_response.status_code = status_code
    mock_response.content = content
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("server error", request=request, response=response)
    )
    client = AsyncMock()
    client.get.return_value = mock_response
    return client


@pytest.mark.unit
class TestHttpErrorMessages:
    """A failed exportImage request reports the reason and the request URL.

    Issue #870 reports NAIP tiles that fail with "HTTP 500" and nothing else.
    The error must carry the ArcGIS message and details when the body has
    them, the body text when it is not JSON, and the exportImage URL in every
    case so the user can reproduce the request.
    """

    EXPORT_URL = (
        "https://example.com/ImageServer/exportImage"
        "?bbox=0.0%2C0.0%2C4096.0%2C4096.0&size=4096%2C4096&format=tiff&f=image"
    )

    @pytest.mark.asyncio
    async def test_http_500_json_error_reports_message_details_and_url(
        self, sample_tile: TileSpec, tmp_path: Path
    ) -> None:
        body = (
            b'{"error":{"code":500,"message":"Unable to complete operation.",'
            b'"details":["Image export failed.","Timeout reading raster."]}}'
        )
        client = _mock_client_with_status(500, body)

        with pytest.raises(ImageServerExtractionError) as exc_info:
            await download_tile(
                "https://example.com/ImageServer", sample_tile, tmp_path / "out.tif", client
            )

        assert str(exc_info.value) == (
            "ArcGIS error for tile tile_0_0: HTTP 500 [500] Unable to complete operation. "
            "Details: Image export failed.; Timeout reading raster. "
            f"Request: {self.EXPORT_URL}"
        )

    @pytest.mark.asyncio
    async def test_http_500_text_body_is_quoted(
        self, sample_tile: TileSpec, tmp_path: Path
    ) -> None:
        client = _mock_client_with_status(500, b"  Internal Server Error\n")

        with pytest.raises(ImageServerExtractionError) as exc_info:
            await download_tile(
                "https://example.com/ImageServer", sample_tile, tmp_path / "out.tif", client
            )

        assert str(exc_info.value) == (
            "Tile download failed (tile_0_0): HTTP 500. Response: Internal Server Error. "
            f"Request: {self.EXPORT_URL}"
        )

    @pytest.mark.asyncio
    async def test_http_500_empty_body_names_request_url(
        self, sample_tile: TileSpec, tmp_path: Path
    ) -> None:
        client = _mock_client_with_status(500, b"")

        with pytest.raises(ImageServerExtractionError) as exc_info:
            await download_tile(
                "https://example.com/ImageServer", sample_tile, tmp_path / "out.tif", client
            )

        assert str(exc_info.value) == (
            "Tile download failed (tile_0_0): HTTP 500 (empty response body). "
            f"Request: {self.EXPORT_URL}"
        )

    @pytest.mark.asyncio
    async def test_http_500_html_body_is_truncated(
        self, sample_tile: TileSpec, tmp_path: Path
    ) -> None:
        client = _mock_client_with_status(500, b"<html>" + b"x" * 500 + b"</html>")

        with pytest.raises(ImageServerExtractionError) as exc_info:
            await download_tile(
                "https://example.com/ImageServer", sample_tile, tmp_path / "out.tif", client
            )

        message = str(exc_info.value)
        assert message.startswith("Tile download failed (tile_0_0): HTTP 500. Response: <html>")
        assert message.endswith(f"Request: {self.EXPORT_URL}")
        assert "</html>" not in message

    @pytest.mark.asyncio
    async def test_http_200_json_error_reports_details_and_url(
        self, sample_tile: TileSpec, tmp_path: Path
    ) -> None:
        body = (
            b'{"error":{"code":400,"message":"The requested image exceeds the size limit.",'
            b'"details":[]}}'
        )
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = body
        mock_response.raise_for_status = MagicMock()
        client = AsyncMock()
        client.get.return_value = mock_response

        with pytest.raises(ImageServerExtractionError) as exc_info:
            await download_tile(
                "https://example.com/ImageServer", sample_tile, tmp_path / "out.tif", client
            )

        assert str(exc_info.value) == (
            "ArcGIS error for tile tile_0_0: HTTP 200 [400] "
            "The requested image exceeds the size limit. "
            f"Request: {self.EXPORT_URL}"
        )


@pytest.mark.unit
class TestFailedTileOutput:
    """The per-tile failure line shows the reason (issue #870)."""

    def test_failed_tile_line_includes_reason(
        self, tmp_path: Path, sample_tile: TileSpec, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from datetime import datetime, timezone

        from portolan_cli.extract.arcgis.imageserver.extractor import (
            _ProcessingStats,
            _update_stats_and_state,
        )
        from portolan_cli.extract.arcgis.imageserver.resume import ImageServerResumeState

        stats = _ProcessingStats()
        state = ImageServerResumeState(
            succeeded_tiles=set(),
            failed_tiles=set(),
            service_url="https://example.com/ImageServer",
            started_at=datetime.now(timezone.utc),
        )

        _update_stats_and_state(
            tile=sample_tile,
            succeeded=False,
            bytes_downloaded=0,
            stats=stats,
            resume_state=state,
            index=0,
            total=1,
            output_dir=tmp_path,
            duration=1.0,
            error_msg="Tile download failed (tile_0_0): HTTP 500 (empty response body)",
            attempts=3,
        )

        captured = capsys.readouterr()
        assert (
            "Tile tile_0_0: failed [1/1]: "
            "Tile download failed (tile_0_0): HTTP 500 (empty response body)"
        ) in captured.err
        assert stats.tiles_failed == 1
        assert stats.tile_results[0].error == (
            "Tile download failed (tile_0_0): HTTP 500 (empty response body)"
        )


# =============================================================================
# License resolution on the raster path (issue #870)
# =============================================================================


@pytest.mark.unit
class TestLicenseResolution:
    """The raster path honors --license and stops early without one.

    Issue #870 runs `extract arcgis <ImageServer> --license CC-BY-4.0`. Before
    the fix the raster path dropped the flag, seeded a TODO placeholder, and
    the add license gate (issue #686) aborted after every tile had downloaded.
    """

    def _run(
        self,
        tmp_path: Path,
        metadata: ImageServerMetadata,
        *,
        license_id: str | None = None,
        license_url: str | None = None,
    ) -> tuple[AsyncMock, Path]:
        """Run extract_imageserver with discovery, download, and init mocked."""
        from portolan_cli.extract.arcgis.imageserver.extractor import _ProcessingStats

        stats = _ProcessingStats()
        mock_tiles = AsyncMock(return_value=stats)
        with (
            patch(
                "portolan_cli.extract.arcgis.imageserver.extractor.discover_imageserver",
                new_callable=AsyncMock,
                return_value=metadata,
            ),
            patch(
                "portolan_cli.extract.arcgis.imageserver.extractor._extract_all_tiles",
                mock_tiles,
            ),
            patch(
                "portolan_cli.extract.arcgis.imageserver.extractor._auto_init_catalog",
                return_value=True,
            ),
        ):
            import asyncio

            asyncio.run(
                extract_imageserver(
                    "https://example.com/ImageServer",
                    tmp_path,
                    config=ExtractionConfig(raw=False),
                    license_id=license_id,
                    license_url=license_url,
                )
            )
        return mock_tiles, tmp_path / ".portolan" / "metadata.yaml"

    def test_license_flag_seeds_metadata_yaml(
        self, tmp_path: Path, small_extent_metadata: ImageServerMetadata
    ) -> None:
        import yaml

        mock_tiles, metadata_path = self._run(
            tmp_path, small_extent_metadata, license_id="CC-BY-4.0"
        )

        assert mock_tiles.await_count == 1
        seeded = yaml.safe_load(metadata_path.read_text())
        assert seeded["license"] == "CC-BY-4.0"
        assert "license_url" not in seeded

    def test_license_flag_with_url_seeds_both(
        self, tmp_path: Path, small_extent_metadata: ImageServerMetadata
    ) -> None:
        import yaml

        _, metadata_path = self._run(
            tmp_path,
            small_extent_metadata,
            license_id="other",
            license_url="https://example.com/terms.html",
        )

        seeded = yaml.safe_load(metadata_path.read_text())
        assert seeded["license"] == "other"
        assert seeded["license_url"] == "https://example.com/terms.html"

    def test_harvested_license_url_seeds_other(
        self, tmp_path: Path, small_extent_metadata: ImageServerMetadata
    ) -> None:
        import yaml

        small_extent_metadata.license_info = (
            "Data licensed under https://creativecommons.org/licenses/by/4.0/"
        )

        _, metadata_path = self._run(tmp_path, small_extent_metadata)

        seeded = yaml.safe_load(metadata_path.read_text())
        assert seeded["license"] == "other"
        assert seeded["license_url"] == "https://creativecommons.org/licenses/by/4.0/"

    def test_missing_license_stops_before_download(
        self, tmp_path: Path, small_extent_metadata: ImageServerMetadata
    ) -> None:
        from portolan_cli.errors import MissingLicenseError

        assert small_extent_metadata.license_info is None

        with pytest.raises(MissingLicenseError, match="publishes no license URL"):
            self._run(tmp_path, small_extent_metadata)

        assert not (tmp_path / "tiles").exists() or not any((tmp_path / "tiles").iterdir())
        assert not (tmp_path / ".portolan" / "imageserver-resume.json").exists()
