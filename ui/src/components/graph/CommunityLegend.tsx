const COMMUNITY_PALETTE = [
  '#a78bfa', '#22d3ee', '#34d399', '#fbbf24', '#f87171',
  '#60a5fa', '#c084fc', '#fb923c', '#2dd4bf', '#e879f9',
  '#818cf8', '#94a3b8', '#f472b6', '#a3e635', '#facc15',
]

interface Props {
  communities: Record<string, number>
}

export function CommunityLegend({ communities }: Props) {
  // Count nodes per community
  const counts = new Map<number, number>()
  for (const cid of Object.values(communities)) {
    counts.set(cid, (counts.get(cid) ?? 0) + 1)
  }

  // Sort by count descending, show top 15
  const sorted = [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15)

  return (
    <div className="absolute bottom-4 left-4 z-10 max-w-xs rounded-lg border border-zinc-800 bg-zinc-900/95 p-3 shadow-xl backdrop-blur-sm">
      <p className="mb-2 text-xs font-medium text-zinc-400">Communities</p>
      <div className="flex flex-wrap gap-2">
        {sorted.map(([cid, count]) => (
          <div key={cid} className="flex items-center gap-1.5 text-xs text-zinc-300">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: COMMUNITY_PALETTE[cid % COMMUNITY_PALETTE.length] }}
            />
            <span className="tabular-nums">#{cid}</span>
            <span className="text-zinc-500">({count})</span>
          </div>
        ))}
      </div>
    </div>
  )
}
