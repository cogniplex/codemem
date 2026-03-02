import { create } from 'zustand'

interface NamespaceState {
  active: string | null
  setActive: (ns: string | null) => void
}

export const useNamespaceStore = create<NamespaceState>((set) => ({
  active: null,
  setActive: (ns) => set({ active: ns }),
}))
