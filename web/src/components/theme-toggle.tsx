import { Check, Monitor, Moon, Sun } from "lucide-react"

import { useTheme } from "@/components/theme-provider"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

const THEME_OPTIONS = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
  { value: "system", label: "System", icon: Monitor },
] as const

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="relative size-11 lg:size-10"
          aria-label="Change theme"
          title="Change theme"
        >
          <Sun
            className="size-5 scale-100 transition-none dark:scale-0"
            strokeWidth={1.75}
          />
          <Moon
            className="absolute size-5 scale-0 transition-none dark:scale-100"
            strokeWidth={1.75}
          />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-36">
        {THEME_OPTIONS.map((option) => {
          const Icon = option.icon
          const selected = theme === option.value
          return (
            <DropdownMenuItem
              key={option.value}
              onClick={() => setTheme(option.value)}
              className="flex items-center justify-between gap-2"
            >
              <span className="flex items-center gap-2">
                <Icon className="size-4" strokeWidth={1.75} />
                <span>{option.label}</span>
              </span>
              {selected ? (
                <Check className="size-4 text-primary" strokeWidth={2} />
              ) : null}
            </DropdownMenuItem>
          )
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
