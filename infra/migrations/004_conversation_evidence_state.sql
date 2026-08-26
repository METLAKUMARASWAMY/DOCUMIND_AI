BEGIN;

ALTER TABLE vision_conversations
  ADD COLUMN IF NOT EXISTS last_resolved_question TEXT NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS last_answer JSONB NOT NULL DEFAULT '{}';

ALTER TABLE vision_conversations
  DROP CONSTRAINT IF EXISTS vision_conversations_pkey;

ALTER TABLE vision_conversations
  ADD PRIMARY KEY (tenant_id, user_id, conversation_id);

COMMIT;
