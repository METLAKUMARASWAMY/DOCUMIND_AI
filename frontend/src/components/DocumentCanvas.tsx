import { Box, Button, FormControlLabel, IconButton, Slider, Switch, ToggleButton, ToggleButtonGroup, Tooltip, Typography } from "@mui/material";
import { ChevronLeft, ChevronRight, Layers2, Maximize2, MoveHorizontal, RotateCw, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
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

export type BaseMode = "original" | "enhanced" | "grayscale" | "binary" | "forensic" | "pdf";
export type Layer = "ocr" | "layout" | "fields" | "tables" | "visual" | "lowConfidence" | "risk" | "tamper";

export interface CanvasOverlay {
  id: string;
  layer: Layer;
  box: BoundingBox;
  label: string;
  variant?: "verified" | "extracted";
}

const DEFAULT_LAYERS: Record<Layer, boolean> = {
  ocr: false,
  layout: false,
  fields: true,
  tables: false,
  visual: false,
  lowConfidence: false,
  risk: false,
  tamper: false,
};

const LAYER_LABELS: Record<Layer, string> = {
  ocr: "OCR",
  layout: "Layout",
  fields: "Fields & attributes",
  tables: "Tables",
  visual: "Visual objects",
  lowConfidence: "Low confidence",
  risk: "Risk indicators",
  tamper: "Tampering",
};

const LEGEND = [
  ["ocr", "OCR"],
  ["layout", "Layout"],
  ["verified", "Verified field"],
  ["fields", "Extracted/discovered field"],
  ["lowConfidence", "Low confidence"],
  ["risk", "Risk indicator"],
  ["tamper", "Possible visual manipulation"],
] as const;

export function DocumentCanvas({ document, page, onPageChange, focus, onRegionClick }: Props) {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [baseMode, setBaseMode] = useState<BaseMode>("original");
  const [overlayOpacity, setOverlayOpacity] = useState(0.8);
  const [layers, setLayers] = useState<Record<Layer, boolean>>(DEFAULT_LAYERS);
  const artifact = document.page_artifacts.find((item) => item.page_number === page);
  const heatmap = document.tamper_analysis.find((item) => item.page_number === page && item.heatmap_path);
  const isPdf = document.filename.toLowerCase().endsWith(".pdf");
  const isNativePdfMode = baseMode === "pdf" && isPdf;
  const imageUrl = useProtectedImage(
    isNativePdfMode
      ? null
      : `/documents/${document.document_id}/pages/${page}/image?variant=${baseMode}`,
  );
  const heatmapUrl = useProtectedImage(
    heatmap ? `/documents/${document.document_id}/artifacts/heatmap/${heatmap.analysis_id}` : null,
  );
  const overlays = useMemo(
    () => buildOverlays(document, page, layers),
    [document, page, layers],
  );
  const width = artifact?.width || 1000;
  const height = artifact?.height || 1400;
  const baseWidth = Math.min(1000, width);
  const stageWidth = baseWidth * zoom;

  useEffect(() => {
    if (focus?.page && focus.page !== page) onPageChange(focus.page);
  }, [focus?.page, onPageChange, page]);

  const updateLayer = (layer: Layer) => {
    setLayers((current) => ({ ...current, [layer]: !current[layer] }));
  };
  const clearLayers = () => setLayers({ ...DEFAULT_LAYERS, fields: false });
  const fitWidth = () => {
    const availableWidth = Math.max(240, (viewportRef.current?.clientWidth || baseWidth) - 56);
    setZoom(clampZoom(availableWidth / baseWidth));
  };
  const fitPage = () => {
    const viewport = viewportRef.current;
    const availableWidth = Math.max(240, (viewport?.clientWidth || baseWidth) - 56);
    const availableHeight = Math.max(320, (viewport?.clientHeight || height) - 56);
    const renderedHeight = baseWidth * (height / width);
    setZoom(clampZoom(Math.min(availableWidth / baseWidth, availableHeight / renderedHeight)));
  };
  const resetView = () => {
    setZoom(1);
    setRotation(0);
  };

  return (
    <Box className="document-canvas-shell">
      <Box className="viewer-toolbar">
        <Box className="viewer-tool-group">
          <Tooltip title="Previous page"><span><IconButton disabled={page <= 1} onClick={() => onPageChange(page - 1)} aria-label="Previous page"><ChevronLeft size={18} /></IconButton></span></Tooltip>
          <Typography variant="body2" className="page-counter">{page} / {document.page_count}</Typography>
          <Tooltip title="Next page"><span><IconButton disabled={page >= document.page_count} onClick={() => onPageChange(page + 1)} aria-label="Next page"><ChevronRight size={18} /></IconButton></span></Tooltip>
          <Tooltip title="Zoom out"><IconButton onClick={() => setZoom((value) => clampZoom(value - 0.15))} aria-label="Zoom out"><ZoomOut size={18} /></IconButton></Tooltip>
          <Typography variant="caption" className="zoom-value">{Math.round(zoom * 100)}%</Typography>
          <Tooltip title="Zoom in"><IconButton onClick={() => setZoom((value) => clampZoom(value + 0.15))} aria-label="Zoom in"><ZoomIn size={18} /></IconButton></Tooltip>
          <Tooltip title="Fit page"><IconButton onClick={fitPage} aria-label="Fit page"><Maximize2 size={18} /></IconButton></Tooltip>
          <Tooltip title="Fit width"><IconButton onClick={fitWidth} aria-label="Fit width"><MoveHorizontal size={18} /></IconButton></Tooltip>
          <Tooltip title="Rotate clockwise"><IconButton onClick={() => setRotation((value) => (value + 90) % 360)} aria-label="Rotate clockwise"><RotateCw size={18} /></IconButton></Tooltip>
          <Tooltip title="Reset view"><IconButton onClick={resetView} aria-label="Reset view"><RotateCcw size={18} /></IconButton></Tooltip>
        </Box>
        <ToggleButtonGroup exclusive size="small" value={baseMode} onChange={(_, value: BaseMode | null) => value && setBaseMode(value)} aria-label="Base image mode">
          <ToggleButton value="original">Original</ToggleButton>
          <ToggleButton value="enhanced">Enhanced</ToggleButton>
          <ToggleButton value="grayscale">Grayscale</ToggleButton>
          <ToggleButton value="binary">Binary</ToggleButton>
          <ToggleButton value="forensic" disabled={!heatmap}>Forensic</ToggleButton>
          {isPdf ? <ToggleButton value="pdf">PDF</ToggleButton> : null}
        </ToggleButtonGroup>
      </Box>
      <Box className="layer-toolbar" aria-label="Evidence overlays">
        {(Object.keys(LAYER_LABELS) as Layer[]).map((layer) => (
          <FormControlLabel
            key={layer}
            control={<Switch size="small" checked={layers[layer]} onChange={() => updateLayer(layer)} />}
            label={LAYER_LABELS[layer]}
          />
        ))}
        <Box className="overlay-opacity">
          <Typography variant="caption">Opacity</Typography>
          <Slider size="small" min={0.15} max={1} step={0.05} value={overlayOpacity} onChange={(_, value) => setOverlayOpacity(value as number)} aria-label="Overlay opacity" />
        </Box>
        <Button size="small" color="inherit" startIcon={<Layers2 size={15} />} onClick={clearLayers}>Clear all overlays</Button>
      </Box>
      <Box className="overlay-legend" aria-label="Overlay legend">
        {LEGEND.map(([kind, label]) => <Box key={kind} className="legend-item"><span className={`legend-swatch legend-${kind}`} /><Typography variant="caption">{label}</Typography></Box>)}
      </Box>
      <Box className="viewer-scroll" ref={viewportRef}>
        {artifact && (isNativePdfMode || imageUrl) ? (
          <Box className="page-stage" style={{ width: `${stageWidth}px`, aspectRatio: `${width} / ${height}`, transform: `rotate(${rotation}deg)` }}>
            {isNativePdfMode ? (
              <Document file={pdfRequest(`/documents/${document.document_id}/source`)} loading={<Box className="viewer-loading"><Typography>Loading PDF page</Typography></Box>} error={<Typography color="error">PDF page could not be rendered.</Typography>}>
                <Page pageNumber={page} width={stageWidth} renderTextLayer={false} renderAnnotationLayer={false} />
              </Document>
            ) : <img src={imageUrl || ""} alt={`${document.filename}, page ${page}, ${baseMode} mode`} />}
            {layers.tamper && baseMode !== "forensic" && heatmapUrl ? <img className="tamper-heatmap" src={heatmapUrl} alt="Possible visual manipulation heatmap" style={{ opacity: overlayOpacity * 0.55 }} /> : null}
            {overlays.map((overlay) => (
              <button
                type="button"
                key={overlay.id}
                className={`evidence-overlay overlay-${overlay.layer}${overlay.variant ? ` overlay-${overlay.variant}` : ""}${focus?.id === overlay.id ? " focused" : ""}`}
                style={{ ...boxStyle(overlay.box, width, height), opacity: overlayOpacity }}
                title={overlay.label}
                aria-label={overlay.label}
                onClick={() => onRegionClick?.({ page, box: overlay.box, id: overlay.id })}
              />
            ))}
            {focus && focus.page === page ? <Box className="focus-overlay" style={boxStyle(focus.box, width, height)} /> : null}
          </Box>
        ) : (
          <Box className="viewer-loading"><Typography>{artifact ? "Loading page image" : "Page artifact unavailable"}</Typography></Box>
        )}
      </Box>
      {!artifact ? <Button size="small" onClick={() => onPageChange(1)}>Return to first page</Button> : null}
    </Box>
  );
}

function useProtectedImage(path: string | null): string | null {
  const [result, setResult] = useState<{ source: string; url: string } | null>(null);
  useEffect(() => {
    if (!path) {
      return;
    }
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

export function buildOverlays(document: VisionDocument, page: number, layers: Record<Layer, boolean>): CanvasOverlay[] {
  const rows: CanvasOverlay[] = [];
  if (layers.ocr) document.ocr_words.filter((item) => item.page === page).forEach((item) => rows.push({ id: item.word_id, layer: "ocr", box: item.bounding_box, label: `OCR: ${item.text} (${Math.round(item.confidence * 100)}%)` }));
  if (layers.layout) document.layout_regions.filter((item) => item.page_number === page).forEach((item) => rows.push({ id: item.region_id, layer: "layout", box: item.bounding_box, label: `${item.region_type}: ${item.text}` }));
  if (layers.tables) document.tables.filter((item) => item.page_number === page).forEach((item) => rows.push({ id: item.table_id, layer: "tables", box: item.bounding_box, label: `Table: ${item.rows.length} row(s)` }));
  if (layers.fields) {
    document.extracted_fields.filter((item) => item.evidence.page_number === page && item.evidence.bounding_box).forEach((item) => rows.push({ id: item.field_id, layer: "fields", variant: verified(item.reviewer_status) ? "verified" : "extracted", box: item.evidence.bounding_box!, label: `${item.display_name || item.name}: ${item.reviewed_value || item.value}` }));
    const verifiedEvidenceIds = new Set((document.verified_facts ?? []).map((fact) => fact.evidence_id));
    (document.generic_fields ?? []).filter((item) => item.page === page).forEach((item) => rows.push({
      id: item.field_id,
      layer: "fields",
      variant: verifiedEvidenceIds.has(item.evidence.evidence_id) ? "verified" : "extracted",
      box: item.evidence.bounding_box || item.value_bbox,
      label: `${item.raw_label || item.normalized_label}: ${item.normalized_value || item.raw_value}`,
    }));
  }
  if (layers.visual) document.visual_detections.filter((item) => item.page_number === page).forEach((item) => rows.push({ id: item.detection_id, layer: "visual", box: item.bounding_box, label: `${item.detection_class} (${Math.round(item.confidence * 100)}%)` }));
  if (layers.lowConfidence) {
    document.ocr_words.filter((item) => item.page === page && item.review_required).forEach((item) => rows.push({ id: `low-${item.word_id}`, layer: "lowConfidence", box: item.bounding_box, label: `Low-confidence OCR: ${item.text}` }));
    document.extracted_fields.filter((item) => item.evidence.page_number === page && item.evidence.bounding_box && (item.confidence === null || item.confidence < 0.7)).forEach((item) => rows.push({ id: `low-${item.field_id}`, layer: "lowConfidence", box: item.evidence.bounding_box!, label: `Low-confidence field: ${item.display_name || item.name}` }));
    (document.generic_fields ?? []).filter((item) => item.page === page && (item.review_required || item.confidence < 0.7)).forEach((item) => rows.push({ id: `low-${item.field_id}`, layer: "lowConfidence", box: item.evidence.bounding_box || item.value_bbox, label: `Low-confidence attribute: ${item.raw_label || item.normalized_label}` }));
  }
  if (layers.risk) document.risk_indicators.forEach((risk) => risk.evidence.filter((item) => item.page_number === page && item.bounding_box).forEach((item) => rows.push({ id: `${risk.risk_id}-${item.evidence_id}`, layer: "risk", box: item.bounding_box!, label: `${risk.title}: ${risk.description}` })));
  if (layers.tamper) document.tamper_analysis.filter((item) => item.page_number === page).forEach((analysis) => analysis.suspicious_regions.forEach((region) => rows.push({ id: region.region_id, layer: "tamper", box: region.bounding_box, label: `${region.manipulation_class} (${Math.round(region.screening_score * 100)} screening score)` })));
  return rows;
}

export function boxStyle(box: BoundingBox, width: number, height: number) {
  return {
    left: `${(box.x1 / width) * 100}%`,
    top: `${(box.y1 / height) * 100}%`,
    width: `${((box.x2 - box.x1) / width) * 100}%`,
    height: `${((box.y2 - box.y1) / height) * 100}%`,
  };
}

function verified(status: string): boolean {
  return status === "accepted" || status === "edited";
}

function clampZoom(value: number): number {
  return Math.min(2.5, Math.max(0.35, value));
}
