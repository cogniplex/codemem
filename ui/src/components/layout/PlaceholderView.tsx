export function PlaceholderView({ title }: { title: string }) {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-zinc-300">{title}</h2>
        <p className="mt-2 text-sm text-zinc-500">Coming soon</p>
      </div>
    </div>
  )
}
