package com.example.orchestration_layer.controller;

import com.example.orchestration_layer.grpc.GrpcQueryClient;
import com.example.orchestration_layer.grpc.QueryProto;
import org.springframework.http.MediaType;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api/v1/corpora")
@CrossOrigin(origins = "*")
public class CorpusController {

    private static final long MAX_FILE_BYTES = 5L * 1024 * 1024;
    private static final int MAX_FILES_PER_REQUEST = 5;
    private final GrpcQueryClient grpcClient;

    public CorpusController(GrpcQueryClient grpcClient) {
        this.grpcClient = grpcClient;
    }

    @PostMapping
    public Map<String, String> createCorpus() {
        return Map.of("corpusId", UUID.randomUUID().toString());
    }

    @GetMapping("/{corpusId}/files")
    public List<Map<String, String>> listFiles(@PathVariable String corpusId) {
        return grpcClient.listCorpusFiles(validateCorpusId(corpusId));
    }

    @PostMapping(value = "/{corpusId}/files", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Map<String, Object> uploadFiles(
            @PathVariable String corpusId,
            @RequestParam("files") List<MultipartFile> files) {
        String validCorpusId = validateCorpusId(corpusId);
        if (files.isEmpty() || files.size() > MAX_FILES_PER_REQUEST) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Upload between 1 and 5 files per request");
        }

        List<Map<String, Object>> results = new ArrayList<>();
        for (MultipartFile file : files) {
            String filename = file.getOriginalFilename();
            try {
                if (file.isEmpty()) {
                    throw new IllegalArgumentException("The selected file is empty");
                }
                if (file.getSize() > MAX_FILE_BYTES) {
                    throw new IllegalArgumentException("Files must be 5 MB or smaller");
                }
                if (filename == null || !filename.toLowerCase().matches(".*\\.(txt|md|pdf|docx|mp3)$")) {
                    throw new IllegalArgumentException("Only .txt, .md, .pdf, .docx and .mp3 files are supported");
                }

                QueryProto.UploadCorpusFileResponse response = grpcClient.uploadCorpusFile(
                        validCorpusId, filename, file.getBytes());
                if (response.getSuccess()) {
                    results.add(Map.of(
                            "success", true,
                            "fileId", response.getFileId(),
                            "filename", response.getFilename(),
                            "chunkCount", response.getChunkCount(),
                            "message", response.getMessage()
                    ));
                } else {
                    results.add(Map.of(
                            "success", false,
                            "filename", filename,
                            "message", response.getMessage()
                    ));
                }
            } catch (IOException | IllegalArgumentException ex) {
                results.add(Map.of(
                        "success", false,
                        "filename", filename == null ? "unknown" : filename,
                        "message", ex.getMessage() == null ? "Upload failed" : ex.getMessage()
                ));
            }
        }
        return Map.of("files", results);
    }

    /** Full text of one uploaded file, for the preview panel. */
    @GetMapping("/{corpusId}/files/{fileId}")
    public Map<String, String> getFile(@PathVariable String corpusId, @PathVariable String fileId) {
        String validFileId = validateUuid(fileId, "file ID");
        var response = grpcClient.getCorpusFile(validateCorpusId(corpusId), validFileId);
        if (!response.getFound()) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND,
                    response.getError().isEmpty() ? "File not found" : response.getError());
        }
        return Map.of("fileId", validFileId, "filename", response.getFilename(), "content", response.getContent());
    }

    /** Removes one uploaded file from the workspace (object storage, graph and vector index). */
    @DeleteMapping("/{corpusId}/files/{fileId}")
    public Map<String, Object> deleteFile(@PathVariable String corpusId, @PathVariable String fileId) {
        var response = grpcClient.deleteCorpusFile(validateCorpusId(corpusId), validateUuid(fileId, "file ID"));
        if (!response.getSuccess()) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, response.getMessage());
        }
        return Map.of("success", true, "message", response.getMessage());
    }

    private String validateCorpusId(String corpusId) {
        return validateUuid(corpusId, "corpus ID");
    }

    private String validateUuid(String value, String label) {
        try {
            UUID parsed = UUID.fromString(value);
            if (!parsed.toString().equalsIgnoreCase(value)) {
                throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Invalid " + label);
            }
            return parsed.toString();
        } catch (ResponseStatusException ex) {
            throw ex;
        } catch (RuntimeException ex) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Invalid " + label, ex);
        }
    }
}
