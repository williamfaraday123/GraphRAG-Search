import { memo, useEffect, useRef, useState } from 'react';

const DAY = 24 * 60 * 60 * 1000;

/** "Today", "Yesterday", "Previous 7 days" or "Older", from the start of today. */
function groupLabel(time, now) {
  const today = new Date(now);
  today.setHours(0, 0, 0, 0);
  const start = today.getTime();
  if (time >= start) return 'Today';
  if (time >= start - DAY) return 'Yesterday';
  if (time >= start - 7 * DAY) return 'Previous 7 days';
  return 'Older';
}

/**
 * One saved chat. Memoised on plain values, so a streaming answer (which changes only the
 * running chat's record) redraws no other row.
 */
const ChatListItem = memo(function ChatListItem({
  id, title, mode, active, running, onSelect, onRename, onDelete,
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(title);
  const inputRef = useRef(null);

  useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  const finish = (save) => {
    setEditing(false);
    const next = draft.trim();
    if (save && next && next !== title) onRename(id, next);
    else setDraft(title);
  };

  if (editing) {
    return (
      <li className="chat-item is-editing">
        <input
          ref={inputRef}
          className="chat-rename-input"
          aria-label="Chat name"
          value={draft}
          maxLength={120}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') finish(true);
            if (e.key === 'Escape') finish(false);
          }}
          onBlur={() => finish(true)}
        />
      </li>
    );
  }

  return (
    <li className={`chat-item ${active ? 'is-active' : ''}`}>
      <button
        type="button"
        className="chat-open"
        aria-current={active ? 'page' : undefined}
        onClick={() => onSelect(id)}
        title={title}
      >
        <span className="chat-title">{title}</span>
        <span className="chat-meta">
          {running && <span className="chat-running" aria-label="Answering">●</span>}
          {mode === 'wiki' ? 'Wiki dump' : 'My files'}
        </span>
      </button>
      <div className="chat-item-actions">
        <button
          type="button"
          className="chat-action"
          aria-label={`Rename ${title}`}
          title="Rename"
          onClick={() => { setDraft(title); setEditing(true); }}
        >
          ✎
        </button>
        <button
          type="button"
          className="chat-action"
          aria-label={`Delete ${title}`}
          title="Delete"
          onClick={() => onDelete(id)}
        >
          ×
        </button>
      </div>
    </li>
  );
});

/**
 * Saved chats, newest first, grouped by day. items: [{ id, title, mode, updatedAt }].
 */
function ChatSidebar({ items, activeId, runningId, open, onNew, onSelect, onRename, onDelete, onClose }) {
  const now = Date.now();
  const groups = [];
  for (const item of items) {
    const label = groupLabel(item.updatedAt, now);
    if (!groups.length || groups[groups.length - 1].label !== label) groups.push({ label, items: [] });
    groups[groups.length - 1].items.push(item);
  }

  return (
    <nav className={`chat-sidebar ${open ? 'is-open' : ''}`} aria-label="Chats">
      <div className="chat-sidebar-top">
        <button type="button" className="new-chat-button" onClick={onNew}>
          + New chat
        </button>
        <button type="button" className="chat-sidebar-close" onClick={onClose} aria-label="Close chats">
          ×
        </button>
      </div>
      {items.length === 0 ? (
        <p className="chat-sidebar-empty">Your chats appear here. They are saved in this browser only.</p>
      ) : (
        <div className="chat-groups">
          {groups.map((group) => (
            <section key={group.label} className="chat-group" aria-label={group.label}>
              <h2 className="chat-group-label">{group.label}</h2>
              <ul className="chat-list">
                {group.items.map((item) => (
                  <ChatListItem
                    key={item.id}
                    id={item.id}
                    title={item.title}
                    mode={item.mode}
                    active={item.id === activeId}
                    running={item.id === runningId}
                    onSelect={onSelect}
                    onRename={onRename}
                    onDelete={onDelete}
                  />
                ))}
              </ul>
            </section>
          ))}
        </div>
      )}
    </nav>
  );
}

export default memo(ChatSidebar);
