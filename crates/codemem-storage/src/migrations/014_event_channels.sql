CREATE TABLE IF NOT EXISTS event_channels (
    id          TEXT PRIMARY KEY,
    namespace   TEXT NOT NULL,
    channel     TEXT NOT NULL,
    direction   TEXT NOT NULL,
    protocol    TEXT DEFAULT '',
    message_schema TEXT DEFAULT '{}',
    description TEXT DEFAULT '',
    handler     TEXT DEFAULT '',
    spec_file   TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_event_channels_ns ON event_channels(namespace);
CREATE INDEX IF NOT EXISTS idx_event_channels_channel ON event_channels(channel);
CREATE INDEX IF NOT EXISTS idx_event_channels_dir ON event_channels(direction);
