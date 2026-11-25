const MARGIN = 16; // room left above what is scrolled to (as scroll-margin-top did)

/**
 * Scrolls the chat area so `el` sits at its top ('start') or middle ('center'), as far as the
 * chat area can scroll, and moves nothing else.
 *
 * Element.scrollIntoView() also scrolls every box around the chat area when the chat area
 * can't go far enough (a new question is the last thing in it), which pushed the whole app
 * up and left a blank strip under it.
 */
export function scrollInChat(container, el, block = 'start') {
  if (!container || !el) return;
  const box = container.getBoundingClientRect();
  const rect = el.getBoundingClientRect();
  const offset = block === 'center'
    ? rect.top - box.top - (box.height - rect.height) / 2
    : rect.top - box.top - MARGIN;
  const max = container.scrollHeight - container.clientHeight;
  const top = Math.max(0, Math.min(container.scrollTop + offset, max));
  if (container.scrollTo) container.scrollTo({ top, behavior: 'smooth' });
  else container.scrollTop = top;
}
