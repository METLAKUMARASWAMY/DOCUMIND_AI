import { Box, Button, FormControlLabel, IconButton, Switch, ToggleButton, ToggleButtonGroup, Tooltip, Typography } from "@mui/material";
import { ChevronLeft, ChevronRight, RotateCw, ZoomIn, ZoomOut } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";

import { pdfRequest, protectedBlob } from "../api/client";
import type { BoundingBox, VisionDocument } from "../types/api";

pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();

export interface FocusRegion {
  page: number;
  box: BoundingBox;
  id?: string;
}

interface Props {
  document: VisionDocument;
  page: number;
  onPageChange: (page: number) => void;
  focus?: FocusRegion | null;
  onRegionClick?: (region: FocusRegion) => void;
}

type Layer = "ocr" | "layout" | "tables" | "fields" | "donut" | "visual" | "risk";

export function DocumentCanvas({ document, page, onPageChange, focus, onRegionClick }: Props) {
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [view, setView] = useState<"processed" | "original" | "pdf">("processed");
  const [layers, setLayers] = useState<Record<Layer, boolean>>({ ocr: true, layout: false, tables: true, fields: true, donut: false, visual: true, risk: true });
  const artifact = document.page_artifacts.find((item) => item.page_number === page);
  const imageUrl = useProtectedImage(`/documents/${document.document_id}/pages/${page}/image?variant=${view === "original" ? "original" : "processed"}`);
  const heatmap = document.tamper_analysis.find((item) => item.page_number === page && item.heatmap_path);
  const heatmapUrl = useProtectedImage(
    heatmap ? `/documents/${document.document_id}/artifacts/heatmap/${heatmap.analysis_id}` : null,
  );
  const overlays = useMemo(() => buildOverlays(document, page, layers), [document, page, layers]);
  const isPdf = document.filename.toLowerCase().endsWith(".pdf");
  const width = artifact?.width || 1000;
  const height = artifact?.height || 1400;

  useEffect(() => {
    if (focus?.page && focus.page !== page) onPageChange(focus.page);
  }, [focus?.page, onPageChange, page]);

  const updateLayer = (layer: Layer) => setLayers((current) => ({ ...current, [layer]: !current[layer] }));
  return (
    <Box className="document-canvas-shell">
      <Box className="viewer-toolbar">
        <Box className="viewer-tool-group">
          <Tooltip title="Previous page"><span><IconButton disabled={page <= 1} onClick={() => onPageChange(page - 1)} aria-label="Previous page"><ChevronLeft size={18} /></IconButton></span></Tooltip>
          <Typography variant="body2" className="page-counter">{page} / {document.page_count}</Typography>
          <Tooltip title="Next page"><span><IconButton disabled={page >= document.page_count} onClick={() => onPageChange(page + 1)} aria-label="Next page"><ChevronRight size={18} /></IconButton></span></Tooltip>
          <Tooltip title="Zoom out"><IconButton onClick={() => setZoom((value) => Math.max(0.55, value - 0.15))} aria-label="Zoom out"><ZoomOut size={18} /></IconButton></Tooltip>
          <Typography variant="caption" className="zoom-value">{Math.round(zoom * 100)}%</Typography>
          <Tooltip title="Zoom in"><IconButton onClick={() => setZoom((value) => Math.min(2.2, value + 0.15))} aria-label="Zoom in"><ZoomIn size={18} /></IconButton></Tooltip>
          <Tooltip title="Rotate clockwise"><IconButton onClick={() => setRotation((value) => (value + 90) % 360)} aria-label="Rotate clockwise"><RotateCw size={18} /></IconButton></Tooltip>
        </Box>
        <ToggleButtonGroup exclusive size="small" value={view} onChange={(_, value) => value && setView(value)} aria-label="Document rendering mode">
          <ToggleButton value="processed">Processed</ToggleButton>
          <ToggleButton value="original">Original</ToggleButton>
          {isPdf ? <ToggleButton value="pdf">PDF</ToggleButton> : null}
        </ToggleButtonGroup>
      </Box>
      <Box className="layer-toolbar" aria-label="Evidence layers">
        {(["ocr", "layout", "tables", "fields", "donut", "visual", "risk"] as Layer[]).map((layer) => (
          <FormControlLabel
            key={layer}
            control={<Switch size="small" checked={layers[layer]} onChange={() => updateLayer(layer)} />}
            label={layer === "fields" ? "LayoutLMv3" : layer === "visual" ? "YOLO" : layer === "tables" ? "Tables" : layer === "donut" ? "Donut" : layer.toUpperCase()}
          />
        ))}
      </Box>
      <Box className="viewer-scroll">
        {view === "pdf" && isPdf ? (
          <Box className="pdf-page" style={{ transform: `rotate(${rotation}deg)` }}>
            <Document file={pdfRequest(`/documents/${document.document_id}/source`)} loading={<Typography>Loading PDF page</Typography>} error={<Typography color="error">PDF page could not be rendered.</Typography>}>
              <Page pageNumber={page} width={Math.min(900, width * zoom)} renderTextLayer={false} renderAnnotationLayer={false} />
            </Document>
          </Box>
        ) : imageUrl ? (
          <Box className="page-stage" style={{ width: `${Math.min(1000, width) * zoom}px`, aspectRatio: `${width} / ${height}`, transform: `rotate(${rotation}deg)` }}>
            <img src={imageUrl} alt={`${document.filename}, page ${page}`} />
            {layers.risk && heatmapUrl ? <img className="tamper-heatmap" src={heatmapUrl} alt="Possible visual manipulation heatmap" /> : null}
            {overlays.map((overlay) => (
              <button
                type="button"
                key={overlay.id}
                className={`evidence-overlay overlay-${overlay.layer}${focus?.id === overlay.id ? " focused" : ""}`}
                style={boxStyle(overlay.box, width, height)}
                title={overlay.label}
                aria-label={overlay.label}
                onClick={() => onRegionClick?.({ page, box: overlay.box, id: overlay.id })}
              />
            ))}
            {focus && focus.page === page ? <Box className="focus-overlay" style={boxStyle(focus.box, width, height)} /> : null}
            {layers.donut ? <DonutComparison document={document} /> : null}
          </Box>
        ) : (
          <Box className="viewer-loading"><Typography>Loading page image</Typography></Box>
        )}
      </Box>
      {!artifact ? <Button size="small" onClick={() => onPageChange(1)}>Return to first page</Button> : null}
    </Box>
  );
}

function useProtectedImage(path: string | null): string | null {
  const [result, setResult] = useState<{ source: string; url: string } | null>(null);
  useEffect(() => {
    if (!path) return;
    let active = true;
    let objectUrl = "";
    void protectedBlob(path)
      .then((blob) => {
        if (!active) return;
        objectUrl = URL.createObjectURL(blob);
        setResult({ source: path, url: objectUrl });
      })
      .catch(() => undefined);
    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [path]);
  return path && result?.source === path ? result.url : null;
}

function buildOverlays(document: VisionDocument, page: number, layers: Record<Layer, boolean>) {
  const rows: Array<{ id: string; layer: Layer; box: BoundingBox; label: string }> = [];
  if (layers.ocr) document.ocr_words.filter((item) => item.page === page).forEach((item) => rows.push({ id: item.word_id, layer: "ocr", box: item.bounding_box, label: `OCR: ${item.text} (${Math.round(item.confidence * 100)}%)` }));
  if (layers.layout) document.layout_regions.filter((item) => item.page_number === page).forEach((item) => rows.push({ id: item.region_id, layer: "layout", box: item.bounding_box, label: `${item.region_type}: ${item.text}` }));
  if (layers.tables) document.tables.filter((item) => item.page_number === page).forEach((item) => rows.push({ id: item.table_id, layer: "tables", box: item.bounding_box, label: `Table: ${item.rows.length} row(s)` }));
  if (layers.fields) document.extracted_fields.filter((item) => item.evidence.page_number === page && item.evidence.bounding_box).forEach((item) => rows.push({ id: item.field_id, layer: "fields", box: item.evidence.bounding_box!, label: `${item.name}: ${item.reviewed_value || item.value}` }));
  if (layers.visual) document.visual_detections.filter((item) => item.page_number === page).forEach((item) => rows.push({ id: item.detection_id, layer: "visual", box: item.bounding_box, label: `${item.detection_class} (${Math.round(item.confidence * 100)}%)` }));
  if (layers.risk) document.risk_indicators.forEach((risk) => risk.evidence.filter((item) => item.page_number === page && item.bounding_box).forEach((item) => rows.push({ id: `${risk.risk_id}-${item.evidence_id}`, layer: "risk", box: item.bounding_box!, label: `${risk.title}: ${risk.description}` })));
  return rows;
}

function DonutComparison({ document }: { document: VisionDocument }) {
  const comparison = document.model_routing.donut_comparison as { disagreements?: unknown[] } | undefined;
  return (
    <Box className="donut-comparison" role="status">
      <Typography variant="caption" fontWeight={700}>Donut comparison</Typography>
      <Typography variant="caption">
        {comparison ? `${comparison.disagreements?.length || 0} field disagreement(s)` : "Not run for this page"}
      </Typography>
    </Box>
  );
}

function boxStyle(box: BoundingBox, width: number, height: number) {
  return {
    left: `${(box.x1 / width) * 100}%`,
    top: `${(box.y1 / height) * 100}%`,
    width: `${((box.x2 - box.x1) / width) * 100}%`,
    height: `${((box.y2 - box.y1) / height) * 100}%`,
  };
}
