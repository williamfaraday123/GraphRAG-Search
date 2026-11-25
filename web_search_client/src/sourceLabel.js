/**
 * Readable name for an object-storage source URI.
 * Uploads (s3://bucket/uploads/<corpus>/<file>/<name>) show the file name; text is stored as
 * "<original>.txt" (report.pdf.txt), reversed like document_text.display_name on the server.
 * Anything else (wiki dump) shows its path without the s3:// prefix.
 */
export function formatSourceLabel(source) {
  if (!source) return 'Unknown source';
  const path = source.startsWith('s3://') ? source.slice('s3://'.length) : source;
  const upload = path.match(/\/uploads\/[^/]+\/[^/]+\/([^/]+)$/);
  if (!upload) return path;
  return upload[1].replace(/(\.(?:txt|md|pdf|docx|mp3))\.txt$/i, '$1');
}

/** The upload file id in a source URI (s3://bucket/uploads/<corpus>/<file>/<name>), or null
 *  for wiki sources. */
export function uploadFileId(source) {
  const match = (source || '').match(/\/uploads\/[^/]+\/([^/]+)\/[^/]+$/);
  return match ? match[1] : null;
}

/** The path without the s3:// prefix, for showing where a source is stored. */
export const sourcePath = (source) => (source || '').replace(/^s3:\/\//, '');

/** Link to this app's full-text page for one source (opened in a new tab from Sources). */
export function sourcePageHref(source) {
  return `${window.location.pathname}?source=${encodeURIComponent(source)}`;
}
