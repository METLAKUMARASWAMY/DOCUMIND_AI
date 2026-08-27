import {
  Alert,
  AlertTitle,
  Box,
  Button,
  Checkbox,
  FormControlLabel,
  LinearProgress,
  MenuItem,
  Step,
  StepLabel,
  Stepper,
  TextField,
  Typography,
} from "@mui/material";
import { FilePlus2, Play, UploadCloud, X } from "lucide-react";
import { useRef, useState } from "react";
import { useNavigate } from "../router";

import { api } from "../api/client";
import { useNotify } from "../components/Notifications";
import { PageHeader } from "../components/PageHeader";
import {
  ACCEPTED_EXTENSIONS,
  MAX_FILES_PER_JOB,
  MAX_UPLOAD_BYTES,
  type Rejection,
  formatBytes,
  validateSelection,
} from "../utils/uploadValidation";

const ACCEPT_ATTRIBUTE = ACCEPTED_EXTENSIONS.map((extension) => `.${extension}`).join(",");

export function UploadWizardPage() {
  const navigate = useNavigate();
  const notify = useNotify();
  const input = useRef<HTMLInputElement | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [rejections, setRejections] = useState<Rejection[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [documentType, setDocumentType] = useState("auto");
  const [advanced, setAdvanced] = useState(false);
  const [requiredSignature, setRequiredSignature] = useState(false);
  const [approvedBank, setApprovedBank] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const activeStep = busy ? 2 : files.length ? 1 : 0;

  const selectFiles = (selected: FileList | null) => {
    if (!selected) return;
    const { accepted, rejected } = validateSelection(Array.from(selected), files);
    if (accepted.length) setFiles((current) => [...current, ...accepted]);
    setRejections(rejected);
    if (rejected.length) {
      notify(
        "warning",
        rejected.length === 1
          ? rejected[0].message
          : `${rejected.length} files were not added. See the details below the drop zone.`,
      );
    }
  };

  const upload = async () => {
    setBusy(true);
    setError("");
    try {
      const result = await api.uploadDocuments(files, {
        document_type_hint: documentType === "auto" ? null : documentType,
        manual_advanced: advanced,
        required_signature: requiredSignature,
        approved_bank_identifier: approvedBank || null,
      });
      notify(
        "success",
        `Job created for ${files.length} document${files.length === 1 ? "" : "s"}.`,
      );
      navigate(`/processing/${result.job_id}`, {
        state: { documentIds: result.document_ids },
      });
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "Upload failed";
      setError(message);
      notify("error", message);
    } finally {
      setBusy(false);
    }
  };

  const totalBytes = files.reduce((sum, file) => sum + file.size, 0);

  return (
    <Box className="page-stack">
      <PageHeader
        title="Upload documents"
        subtitle="Create an asynchronous visual intelligence job."
      />
      <Stepper activeStep={activeStep} className="compact-stepper">
        {["Select files", "Configure", "Process"].map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>
      {error ? (
        <Alert severity="error" onClose={() => setError("")}>
          <AlertTitle>The job could not be created</AlertTitle>
          {error}
        </Alert>
      ) : null}
      <Box className="upload-layout">
        <Box
          className={`surface upload-dropzone${dragActive ? " is-drag-active" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragActive(false);
            selectFiles(event.dataTransfer.files);
          }}
        >
          <UploadCloud size={32} aria-hidden />
          <Typography variant="h3">Drop documents here</Typography>
          <Typography color="text.secondary">
            {ACCEPTED_EXTENSIONS.join(", ").toUpperCase()} · up to {formatBytes(MAX_UPLOAD_BYTES)}{" "}
            each · {MAX_FILES_PER_JOB} per job
          </Typography>
          <input
            ref={input}
            type="file"
            multiple
            hidden
            accept={ACCEPT_ATTRIBUTE}
            aria-label="Choose documents to upload"
            onChange={(event) => {
              selectFiles(event.target.files);
              event.target.value = "";
            }}
          />
          <Button
            variant="outlined"
            startIcon={<FilePlus2 size={17} />}
            onClick={() => input.current?.click()}
          >
            Browse files
          </Button>
        </Box>
        <Box className="surface config-panel">
          <Typography variant="h3">Processing profile</Typography>
          <TextField
            select
            label="Document type"
            value={documentType}
            onChange={(event) => setDocumentType(event.target.value)}
            helperText="Automatic classification is used unless you pin a type."
            fullWidth
          >
            {[
              ["auto", "Automatic classification"],
              ["invoice", "Invoice"],
              ["purchase_order", "Purchase order"],
              ["contract", "Contract"],
              ["receipt", "Receipt"],
              ["bank_document", "Bank document"],
              ["insurance_document", "Insurance document"],
              ["form", "Form"],
              ["report", "Report"],
            ].map(([value, label]) => (
              <MenuItem key={value} value={value}>
                {label}
              </MenuItem>
            ))}
          </TextField>
          <FormControlLabel
            control={
              <Checkbox
                checked={advanced}
                onChange={(event) => setAdvanced(event.target.checked)}
              />
            }
            label="Request advanced parser"
          />
          <FormControlLabel
            control={
              <Checkbox
                checked={requiredSignature}
                onChange={(event) => setRequiredSignature(event.target.checked)}
              />
            }
            label="Signature required"
          />
          <TextField
            label="Approved bank identifier"
            value={approvedBank}
            onChange={(event) => setApprovedBank(event.target.value)}
            helperText="Optional. Bank details that differ from this value are flagged for review."
            fullWidth
          />
        </Box>
      </Box>
      {rejections.length ? (
        <Alert severity="warning" onClose={() => setRejections([])}>
          <AlertTitle>
            {rejections.length} file{rejections.length === 1 ? " was" : "s were"} not added
          </AlertTitle>
          <Box component="ul" className="rejection-list">
            {rejections.map((rejection) => (
              <li key={`${rejection.file.name}-${rejection.reason}`}>{rejection.message}</li>
            ))}
          </Box>
        </Alert>
      ) : null}
      {files.length ? (
        <Box className="surface file-queue">
          <Typography variant="h3">
            Files ({files.length}) · {formatBytes(totalBytes)}
          </Typography>
          {files.map((file, index) => (
            <Box className="file-row" key={`${file.name}-${file.size}-${file.lastModified}`}>
              <Box>
                <Typography fontWeight={650}>{file.name}</Typography>
                <Typography variant="caption" color="text.secondary">
                  {formatBytes(file.size)}
                </Typography>
              </Box>
              <Button
                color="inherit"
                size="small"
                disabled={busy}
                startIcon={<X size={15} />}
                aria-label={`Remove ${file.name}`}
                onClick={() =>
                  setFiles((current) => current.filter((_, itemIndex) => itemIndex !== index))
                }
              >
                Remove
              </Button>
            </Box>
          ))}
          {busy ? (
            <Box>
              <LinearProgress aria-label="Creating processing job" />
              <Typography variant="caption" color="text.secondary">
                Transferring documents and creating the job. Processing progress appears on the next
                screen.
              </Typography>
            </Box>
          ) : null}
          <Box className="form-actions">
            <Button
              variant="contained"
              startIcon={<Play size={17} />}
              disabled={!files.length || busy}
              onClick={() => void upload()}
            >
              {busy ? "Creating job" : "Start processing"}
            </Button>
            <Button color="inherit" disabled={busy} onClick={() => setFiles([])}>
              Clear queue
            </Button>
          </Box>
        </Box>
      ) : null}
    </Box>
  );
}
