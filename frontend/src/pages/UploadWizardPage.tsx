import { Alert, Box, Button, Checkbox, FormControlLabel, LinearProgress, MenuItem, Step, StepLabel, Stepper, TextField, Typography } from "@mui/material";
import { FilePlus2, Play, UploadCloud, X } from "lucide-react";
import { useRef, useState } from "react";
import { useNavigate } from "../router";

import { api } from "../api/client";
import { PageHeader } from "../components/PageHeader";

const accepted = ["PDF", "TXT", "PNG", "JPG", "JPEG", "TIFF"];

export function UploadWizardPage() {
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [documentType, setDocumentType] = useState("auto");
  const [advanced, setAdvanced] = useState(false);
  const [requiredSignature, setRequiredSignature] = useState(false);
  const [approvedBank, setApprovedBank] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const activeStep = files.length ? 1 : 0;

  const selectFiles = (selected: FileList | null) => {
    if (!selected) return;
    setFiles((current) => [...current, ...Array.from(selected)].slice(0, 20));
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
      navigate(`/processing/${result.job_id}`, {
        state: { documentIds: result.document_ids },
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Box className="page-stack">
      <PageHeader title="Upload documents" subtitle="Create an asynchronous visual intelligence job." />
      <Stepper activeStep={activeStep} className="compact-stepper">
        {["Select files", "Configure", "Process"].map((label) => <Step key={label}><StepLabel>{label}</StepLabel></Step>)}
      </Stepper>
      {error ? <Alert severity="error">{error}</Alert> : null}
      <Box className="upload-layout">
        <Box className="surface upload-dropzone" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); selectFiles(event.dataTransfer.files); }}>
          <UploadCloud size={32} />
          <Typography variant="h3">Drop documents here</Typography>
          <Typography color="text.secondary">{accepted.join(", ")} up to 25 MB each</Typography>
          <input
            ref={input}
            type="file"
            multiple
            hidden
            accept=".pdf,.txt,.png,.jpg,.jpeg,.tif,.tiff"
            onChange={(event) => selectFiles(event.target.files)}
          />
          <Button variant="outlined" startIcon={<FilePlus2 size={17} />} onClick={() => input.current?.click()}>
            Browse files
          </Button>
        </Box>
        <Box className="surface config-panel">
          <Typography variant="h3">Processing profile</Typography>
          <TextField select label="Document type" value={documentType} onChange={(event) => setDocumentType(event.target.value)} fullWidth>
            {[
              ["auto", "Automatic classification"], ["invoice", "Invoice"], ["purchase_order", "Purchase order"],
              ["contract", "Contract"], ["receipt", "Receipt"], ["bank_document", "Bank document"],
              ["insurance_document", "Insurance document"], ["form", "Form"], ["report", "Report"],
            ].map(([value, label]) => <MenuItem key={value} value={value}>{label}</MenuItem>)}
          </TextField>
          <FormControlLabel control={<Checkbox checked={advanced} onChange={(event) => setAdvanced(event.target.checked)} />} label="Request advanced parser" />
          <FormControlLabel control={<Checkbox checked={requiredSignature} onChange={(event) => setRequiredSignature(event.target.checked)} />} label="Signature required" />
          <TextField label="Approved bank identifier" value={approvedBank} onChange={(event) => setApprovedBank(event.target.value)} fullWidth />
        </Box>
      </Box>
      {files.length ? (
        <Box className="surface file-queue">
          <Typography variant="h3">Files ({files.length})</Typography>
          {files.map((file, index) => (
            <Box className="file-row" key={`${file.name}-${file.lastModified}`}>
              <Box><Typography fontWeight={650}>{file.name}</Typography><Typography variant="caption" color="text.secondary">{(file.size / 1024 / 1024).toFixed(2)} MB</Typography></Box>
              <Button color="inherit" size="small" startIcon={<X size={15} />} onClick={() => setFiles((current) => current.filter((_, itemIndex) => itemIndex !== index))}>Remove</Button>
            </Box>
          ))}
          {busy ? <LinearProgress /> : null}
          <Box className="form-actions">
            <Button variant="contained" startIcon={<Play size={17} />} disabled={!files.length || busy} onClick={() => void upload()}>
              {busy ? "Creating job" : "Start processing"}
            </Button>
          </Box>
        </Box>
      ) : null}
    </Box>
  );
}
