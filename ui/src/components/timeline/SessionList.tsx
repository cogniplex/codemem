import { useState } from 'react'
import { ChevronDown, ChevronRight, Clock } from 'lucide-react'
import { useSessions } from '../../api/hooks'
import type { SessionResponse } from '../../api/types'
import { useNamespaceStore } from '../../stores/namespace'

function SessionRow({ session }: { session: SessionResponse }) {
  const [expanded, setExpanded] = useState(false)
  const isActive = !session.ended_at

  return (
    <div className="border-b border-zinc-800/50 last:border-b-0">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-zinc-800/30"
      >
        {expanded ? (
          <ChevronDown size={14} className="shrink-0 text-zinc-500" />
        ) : (
          <ChevronRight size={14} className="shrink-0 text-zinc-500" />
        )}
        <code className="shrink-0 text-xs text-zinc-400">
          {session.id.slice(0, 8)}
        </code>
        {session.namespace && (
          <span className="max-w-[140px] shrink-0 truncate rounded bg-zinc-800 px-1.5 py-0.5 text-xs text-zinc-400" title={session.namespace}>
            {session.namespace}
          </span>
        )}
        {isActive ? (
          <span className="shrink-0 rounded bg-emerald-500/20 px-1.5 py-0.5 text-xs font-medium text-emerald-400">
            Active
          </span>
        ) : null}
        <span className="ml-auto shrink-0 text-xs tabular-nums text-zinc-500">
          {session.memory_count} memories
        </span>
        <time className="shrink-0 text-xs text-zinc-600">
          {new Date(session.started_at).toLocaleString()}
        </time>
      </button>
      {expanded && (
        <div className="border-t border-zinc-800/30 bg-zinc-900/50 px-4 py-3 pl-10">
          <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs">
            <div>
              <dt className="text-zinc-500">Session ID</dt>
              <dd className="break-all font-mono text-zinc-300">{session.id}</dd>
            </div>
            <div>
              <dt className="text-zinc-500">Namespace</dt>
              <dd className="text-zinc-300">{session.namespace ?? 'default'}</dd>
            </div>
            <div>
              <dt className="text-zinc-500">Started</dt>
              <dd className="text-zinc-300">
                {new Date(session.started_at).toLocaleString()}
              </dd>
            </div>
            <div>
              <dt className="text-zinc-500">Ended</dt>
              <dd className="text-zinc-300">
                {session.ended_at
                  ? new Date(session.ended_at).toLocaleString()
                  : 'Still active'}
              </dd>
            </div>
            <div>
              <dt className="text-zinc-500">Memories</dt>
              <dd className="text-zinc-300">{session.memory_count}</dd>
            </div>
            {session.summary && (
              <div className="col-span-2">
                <dt className="text-zinc-500">Summary</dt>
                <dd className="mt-1 break-words text-zinc-300">{session.summary}</dd>
              </div>
            )}
          </dl>
        </div>
      )}
    </div>
  )
}

export function SessionList() {
  const namespace = useNamespaceStore((s) => s.active)
  const { data: sessions, isLoading } = useSessions({
    namespace: namespace ?? undefined,
  })

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-850">
      <div className="flex items-center gap-2 border-b border-zinc-800 px-5 py-3">
        <Clock size={16} className="text-zinc-400" />
        <h3 className="text-sm font-medium text-zinc-300">Sessions</h3>
        {sessions && (
          <span className="text-xs text-zinc-500">({sessions.length})</span>
        )}
      </div>
      <div>
        {isLoading ? (
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="flex items-center gap-3 px-5 py-3">
              <span className="h-4 w-16 animate-pulse rounded bg-zinc-800" />
              <span className="h-4 w-24 animate-pulse rounded bg-zinc-800" />
              <span className="ml-auto h-4 w-20 animate-pulse rounded bg-zinc-800" />
            </div>
          ))
        ) : !sessions || sessions.length === 0 ? (
          <p className="px-5 py-8 text-center text-sm text-zinc-500">
            No sessions found
          </p>
        ) : (
          sessions.map((s) => <SessionRow key={s.id} session={s} />)
        )}
      </div>
    </div>
  )
}
