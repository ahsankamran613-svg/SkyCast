# SkyCast (AgentWeather) Frontend Design Spec

Design system for the SkyCast dashboard: Lahore weather and air quality forecasting, 4-agent MLOps layer, light and dark mode.

- **Palette:** Sky Slate (cool blue, classic weather and broadcast feel)
- **Stack:** React + Vite, Tailwind CSS v4, shadcn/ui, Recharts
- **Modes:** Light, Dark, and System (default), all tokens defined for both
- **Base unit:** 4px spacing grid, 16px root font size

---

## 1. Design principles

1. **Data first.** Numbers (temperature, AQI, RMSE) are the largest things on screen. Everything else supports them.
2. **One memorable element.** The hero "current conditions" panel and AQI gauge carry the personality. Everything around them stays quiet.
3. **Color means something.** Blue is interface. Orange, violet, teal, amber, and slate identify metrics. Green-to-maroon is only for air quality. Red, amber, and sky are only for alert severity.
4. **Never color alone.** Every AQI level and alert also has a text label and an icon.
5. **Radius follows hierarchy.** Bigger surfaces get bigger radii; small controls get small radii. No single radius on everything.
6. **Tokens only.** No hex values inside components or chart code. Everything reads from CSS variables, so light and dark mode work everywhere.

---

## 2. Color

### 2.1 Core UI tokens

| Token | Light | Dark | Used for |
|---|---|---|---|
| `--background` | `#F6F8FB` | `#0B1220` | Page background |
| `--foreground` | `#0F172A` | `#E6EDF7` | Primary text |
| `--card` | `#FFFFFF` | `#111A2E` | Cards, panels |
| `--popover` | `#FFFFFF` | `#111A2E` | Menus, tooltips, dialogs |
| `--muted` | `#EEF2F7` | `#142038` | Subtle fills, table stripes |
| `--muted-foreground` | `#64748B` | `#8FA0BC` | Secondary text, axis labels |
| `--secondary` | `#E8EEF6` | `#17233D` | Secondary buttons, chips |
| `--border` | `#E2E8F0` | `#1E2A44` | Borders, dividers, chart grid |
| `--primary` | `#0369A1` | `#38BDF8` | Buttons, links, active nav |
| `--primary-foreground` | `#FFFFFF` | `#0B1220` | Text on primary |
| `--accent` | `#E0F2FE` | `#123349` | Hover and selected backgrounds |
| `--accent-foreground` | `#075985` | `#7DD3FC` | Text on accent |
| `--ring` | `#0EA5E9` | `#38BDF8` | Focus ring |
| `--destructive` | `#B91C1C` | `#F87171` | Destructive actions |
| `--sidebar` | `#FFFFFF` | `#0D1526` | Sidebar background |

**Rules**

- Use `--foreground`, not `--muted-foreground`, for text sitting on `--muted` or `--secondary` surfaces (muted text only meets 4.5:1 on `--background` and `--card`).
- Dark mode is never pure black or pure white. Background is navy-tinted, text is `#E6EDF7`.
- In dark mode, separate surfaces with borders instead of shadows.

### 2.2 Air quality (AQI) scale

Use the **fill** for gauge arcs, bars, dots, and chart zone bands. Use **text** and **tint** for badges.

| Level | Range | Light fill | Light text | Light tint | Dark fill | Dark text | Dark tint |
|---|---|---|---|---|---|---|---|
| Good | 0–50 | `#059669` | `#065F46` | `#D1FAE5` | `#34D399` | `#34D399` | `rgba(52,211,153,.15)` |
| Moderate | 51–100 | `#CA8A04` | `#854D0E` | `#FEF9C3` | `#FACC15` | `#FACC15` | `rgba(250,204,21,.15)` |
| Unhealthy for sensitive groups | 101–150 | `#EA580C` | `#9A3412` | `#FFEDD5` | `#FB923C` | `#FB923C` | `rgba(251,146,60,.15)` |
| Unhealthy | 151–200 | `#DC2626` | `#991B1B` | `#FEE2E2` | `#F87171` | `#F87171` | `rgba(248,113,113,.15)` |
| Very unhealthy | 201–300 | `#9333EA` | `#6B21A8` | `#F3E8FF` | `#C084FC` | `#C084FC` | `rgba(192,132,252,.15)` |
| Hazardous | 301+ | `#9F1239` | `#881337` | `#FFE4E6` | `#E11D48` | `#FDA4AF` | `rgba(225,29,72,.22)` |

### 2.3 Alert severity

| Severity | Light text | Light tint | Dark text | Dark tint | Icon |
|---|---|---|---|---|---|
| Info | `#0369A1` | `#E0F2FE` | `#38BDF8` | `rgba(56,189,248,.12)` | `Info` |
| Warning | `#B45309` | `#FEF3C7` | `#FBBF24` | `rgba(251,191,36,.12)` | `TriangleAlert` |
| Critical | `#B91C1C` | `#FEE2E2` | `#F87171` | `rgba(248,113,113,.12)` | `OctagonAlert` |

### 2.4 Metric chart colors

| Metric | Light | Dark | Token |
|---|---|---|---|
| Temperature | `#EA580C` | `#FB923C` | `--chart-1` |
| AQI | `#7C3AED` | `#A78BFA` | `--chart-2` |
| Precipitation | `#2563EB` | `#60A5FA` | `--chart-3` |
| Humidity | `#0D9488` | `#2DD4BF` | `--chart-4` |
| UV index | `#CA8A04` | `#FACC15` | `--chart-5` |
| Wind | `#64748B` | `#94A3B8` | `--chart-6` |

Chart helpers: grid `var(--border)`, axis text `var(--muted-foreground)`, forecast band = metric color at 15% opacity.

### 2.5 Model family colors (Models tab)

| Model | Light | Dark | Line style |
|---|---|---|---|
| SARIMAX | `#7C3AED` | `#A78BFA` | Solid |
| Random Forest | `#16A34A` | `#4ADE80` | Solid |
| Ridge | `#D97706` | `#FBBF24` | Solid |
| LSTM | `#E11D48` | `#FB7185` | Solid |
| Naive 24h baseline | `#64748B` | `#94A3B8` | Dashed (`6 4`) |

### 2.6 Explainability and EDA

| Use | Light | Dark |
|---|---|---|
| SHAP/LIME pushes forecast up | `#EA580C` | `#FB923C` |
| SHAP/LIME pushes forecast down | `#0369A1` | `#38BDF8` |
| Correlation matrix (negative → neutral → positive) | `#0369A1` → `#F8FAFC` → `#EA580C` | `#38BDF8` → `#1E2A44` → `#FB923C` |
| Diurnal heatmap (low → high) | `#F0F9FF`, `#BAE6FD`, `#38BDF8`, `#0284C7`, `#0C4A6E` | `#0F1B30`, `#0C4A6E`, `#0284C7`, `#38BDF8`, `#E0F2FE` |

### 2.7 Drift status

| State | Token |
|---|---|
| No drift | `--aqi-good` fill, with `CircleCheck` icon |
| Drift detected, retrain triggered | `--alert-warning`, with `TriangleAlert` icon |
| Check failed | `--alert-critical`, with `OctagonAlert` icon |

---

## 3. Typography

| Role | Family | Weights |
|---|---|---|
| Display and big numbers | **Barlow Semi Condensed** | 500, 600, 700 |
| Interface and body | **Public Sans** | 400, 500, 600 |

Barlow Semi Condensed gives the large readouts a broadcast lower-third feel without taking much width. Public Sans keeps dense tables and logs legible. Both are on Google Fonts and load in `web/index.html` via `<link>` (or bundled via `@fontsource/barlow-semi-condensed` and `@fontsource/public-sans`). Fallback stack: `ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif`.

### 3.1 Type scale

| Style | Font | Size / line height (desktop) | Mobile | Weight | Tracking | Used for |
|---|---|---|---|---|---|---|
| `hero` | Barlow | 72 / 72 px | 56 / 56 | 600 | -0.02em | Current temperature |
| `display` | Barlow | 48 / 52 | 40 / 44 | 600 | -0.01em | AQI value, large KPIs |
| `stat` | Barlow | 32 / 36 | 28 / 32 | 600 | -0.01em | Metric card values |
| `h1` | Barlow | 30 / 36 | 26 / 32 | 600 | 0 | Page title |
| `h2` | Barlow | 22 / 28 | 20 / 26 | 600 | 0 | Section title |
| `h3` | Public Sans | 16 / 24 | 16 / 24 | 600 | 0 | Card title |
| `body` | Public Sans | 16 / 24 | 16 / 24 | 400 | 0 | Default text |
| `body-sm` | Public Sans | 14 / 20 | 14 / 20 | 400 | 0 | Table cells, descriptions |
| `label` | Public Sans | 13 / 18 | 13 / 18 | 500 | 0 | Field and metric labels |
| `caption` | Public Sans | 12 / 16 | 12 / 16 | 400 | 0 | Axis ticks, timestamps, tooltips |
| `bulletin` | Public Sans | 17 / 28 | 16 / 26 | 400 | 0 | Weather report body |

### 3.2 Type rules

- All numbers in tables, charts, stat cards, and logs use `font-variant-numeric: tabular-nums` so columns don't jitter on update.
- Units (°C, km/h, %) sit next to the value at 14px / weight 500 in `--muted-foreground`.
- Body line length stays under 80 characters. The weather bulletin is capped at `68ch`.
- Sentence case everywhere. No all-caps labels, no tracked-out eyebrow text above headings.
- Use weight (600) for emphasis, not color.

---

## 4. Spacing and layout

### 4.1 Spacing scale (4px base)

`0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64` px, mapped to Tailwind `0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16`.

| Use | Desktop | Mobile |
|---|---|---|
| Page padding | 24 (32 at ≥1536) | 16 |
| Gap between cards | 16 | 12 |
| Gap between page sections | 32 | 24 |
| Card padding | 20 | 16 |
| Hero panel padding | 28 | 20 |
| Inline gap (icon + text) | 8 | 8 |
| Form field gap | 16 | 16 |

### 4.2 Breakpoints (Tailwind defaults)

| Name | Min width | Layout |
|---|---|---|
| base | 0 | Single column, bottom nav, 4-col grid |
| `sm` | 640 | Single column, 2 stat cards per row |
| `md` | 768 | 8-col grid, 3 stat cards per row |
| `lg` | 1024 | Sidebar appears (72px icon rail) |
| `xl` | 1280 | Full 248px sidebar, 12-col grid, 6 stat cards per row |
| `2xl` | 1536 | Same layout, wider page padding |

### 4.3 App shell dimensions

| Element | Size |
|---|---|
| Sidebar, expanded (≥1280) | 248px wide |
| Sidebar, icon rail (1024–1279) | 72px wide |
| Header height | 64px desktop, 56px mobile |
| Bottom nav (below 1024) | 64px + `env(safe-area-inset-bottom)` |
| Content max width | 1440px, centered |
| Report/text pages max width | 800px |

### 4.4 Navigation (6 tabs)

Overview, Forecasts, Models, EDA, Agents and drift, Technical report.

- **Desktop:** left sidebar, items 40px tall, 12px horizontal padding, 20px icons, 12px gap, 8px radius. Active item uses `--sidebar-accent` fill and `--sidebar-accent-foreground` text.
- **Mobile:** bottom nav with 4 tabs (Overview, Forecasts, Models, Agents) plus a "More" sheet (EDA, Technical report). Each item is at least 60×56px, icon 22px, label 12px.
- **Header:** city name and "updated X min ago" on the left. Theme toggle and refresh on the right.

---

## 5. Shape, borders, elevation

### 5.1 Radius hierarchy

| Token | Value | Used for |
|---|---|---|
| `--radius-sm` | 6px | Tooltips, small chips, table cell highlights |
| `--radius-md` | 8px | Buttons, inputs, selects, sidebar items |
| `--radius-lg` | 12px | Cards, alert banners, dialogs |
| `--radius-xl` | 16px | Large panels (forecast chart, report) |
| `--radius-2xl` | 20px | Hero current-conditions panel only |
| `--radius-full` | 9999px | Badges, toggles, avatars |

### 5.2 Borders

- Default: 1px `--border`.
- Alert banners: 1px `--border` plus a 4px left edge in the severity color.
- Selected/active cards: 1px `--primary`.
- No double borders. A card inside a card uses `--muted` fill and no border.

### 5.3 Elevation

| Level | Light | Dark |
|---|---|---|
| Card | `0 1px 2px rgb(15 23 42 / 0.04)` | none (border only) |
| Popover, menu, tooltip | `0 8px 24px rgb(15 23 42 / 0.12)` | `0 8px 24px rgb(0 0 0 / 0.5)` plus border |
| Dialog | `0 16px 48px rgb(15 23 42 / 0.18)` | `0 16px 48px rgb(0 0 0 / 0.6)` plus border |

### 5.4 Focus

Every interactive element gets a visible keyboard focus: `outline: 2px solid var(--ring); outline-offset: 2px`. Never remove outlines without replacing them.

---

## 6. Component sizes

### 6.1 Controls

| Component | Size |
|---|---|
| Button, small | 32px high, 12px horizontal padding, 14px text |
| Button, default | 40px high, 16px padding, 14px text |
| Button, large | 48px high, 20px padding, 16px text |
| Icon button | 40×40 (44×44 on mobile) |
| Input, select | 40px high, 12px padding, 14px text |
| Theme toggle | 40×40 icon button, 20px icon, menu with Light / Dark / System |
| Tabs list | 40px high; trigger 16px horizontal padding, 14px text |
| Segmented toggle (SHAP / LIME) | 36px high, pill-shaped |
| Switch | 40×22 track, 18px thumb |
| Minimum touch target | 44×44px on mobile |

### 6.2 Icons

Lucide, stroke width 1.75. Sizes: 16 (inline with text), 20 (default, buttons, sidebar), 24 (empty states), 22 (bottom nav).

### 6.3 Badges

| Type | Size |
|---|---|
| Standard badge | 24px high, 8px horizontal padding, 12px text, weight 500, pill |
| AQI badge (large) | 32px high, 12px horizontal padding, 14px text, weight 600, pill, level icon plus name |

### 6.4 Cards

| Card | Size and layout |
|---|---|
| Hero (current conditions) | Min height 220px, padding 28, `--radius-2xl`, `--accent` tinted background |
| Metric stat card | Min height 132px, padding 20, `--radius-lg`; label (13px), value (`stat`) with unit, 40px sparkline at the bottom |
| Chart card | Padding 20, `--radius-xl`; title row 32px, then chart |
| Alert banner | Min height 56px, padding 12×16, `--radius-lg`, 20px icon, 4px severity left edge |
| Audience advisory (General / Sensitive / Outdoor workers) | Three equal columns on desktop, stacked on mobile, padding 16 |
| Weather bulletin | Padding 24, `--radius-xl`, text max `68ch` |

### 6.5 Data displays

| Component | Size |
|---|---|
| AQI gauge (semicircle) | 240×140 desktop, 200×120 mobile; arc stroke 14px, rounded caps; value in `display` style centered |
| Wind compass | 120px diameter desktop, 96px mobile; arrow rotated by wind-from degrees; speed in `stat` style beside it |
| Sparkline | 40px high, 2px stroke, no axes (plots the last 24 observed hours) |
| Data table | Header 40px, row 44px (52px on mobile), cell padding 12×16, sticky header, hover `--muted` |
| Leaderboard bar | 28px high bars, 12px gap, best model row highlighted with `--accent` |
| Log row (agent calls) | Min height 48px, status badge at left, latency right-aligned, tabular numbers |
| Tooltip | Padding 8×12, 12px text, `--radius-sm`, max width 280px, popover elevation |

### 6.6 Charts

| Chart | Height desktop | Height mobile |
|---|---|---|
| Main forecast chart | 320px | 240px |
| Small multiples (6 metrics) | 160px | 140px |
| Forecast vs actual history | 300px | 240px |
| Model comparison lines | 320px | 240px |
| SHAP/LIME bars | 28px per bar, 10 bars | same |
| Diurnal heatmap | Cells 28×28px (min 20px on mobile) | scroll horizontally |
| Correlation matrix | Cells 48×48px (min 36px on mobile) | scroll horizontally |

| Chart rule | Value |
|---|---|
| Line stroke | 2px (hover 2.5px) |
| Forecast line | Same color as observed, dashed `6 4` |
| Confidence band | Metric color at 15% opacity |
| Dots | Hidden by default, 5px radius on hover |
| Grid | Horizontal only, 1px, `--border` |
| Axis ticks | 12px `--muted-foreground`, no tick lines |
| "Now" marker | 1px vertical line, `--foreground` at 40% opacity, label "Now" (Overview chart shows 12h observed lead-in + 24h forecast; Forecasts tab shows 24h observed + 24h forecast with `12h / 24h` toggle) |
| AQI zone bands | Chart background bands using AQI fill colors at 8% opacity |
| Legend | Below chart, 12px text, 8px color swatch (pill) |
| Entry animation | Line draws once on first load (600ms). No animation on refetch. |

Every chart needs an `aria-label` and a "Show as table" toggle for screen readers.

---

## 7. Page layouts

### 7.1 Overview (desktop, ≥1280)

```
┌──────────┬────────────────────────────────────────────────────────────┐
│ Sidebar  │ Header: Lahore, updated 12 min ago           [Theme][Refresh]│
│ 248px    ├────────────────────────────────────────────────────────────┤
│          │ Alert banner (only when a hazard is active)                 │
│          ├────────────────────────────────┬───────────────────────────┤
│          │ Hero: temperature, feels like, │ AQI gauge + pollutants     │
│          │ condition, wind compass        │ (PM2.5, PM10, O3, NO2)     │
│          │ (7 cols)                       │ (5 cols)                   │
│          ├──────┬──────┬──────┬──────┬────┴─┬─────────────────────────┤
│          │ Temp │ Humid│ Rain │ UV   │ Wind │ AQI   (6 stat cards, 2   │
│          │      │      │      │      │      │ cols each)               │
│          ├────────────────────────────────┬───────────────────────────┤
│          │ 24h forecast chart             │ Weather bulletin           │
│          │ (metric tabs, 8 cols)          │ (4 cols)                   │
│          ├────────────────────────────────┴───────────────────────────┤
│          │ Advisories: General public | Sensitive groups | Outdoor work│
└──────────┴────────────────────────────────────────────────────────────┘
```

On tablet, hero and AQI stack, stat cards go 3 per row, and the chart and bulletin stack. On mobile, everything is a single column in this order: alert banner, hero, AQI gauge, stat cards (2 per row), forecast chart, bulletin, advisories.

### 7.2 Other tabs

| Tab | Layout |
|---|---|
| **Forecasts** | Metric selector tabs on top, large forecast chart (24h trajectory with band), forecast vs actual history chart with naive baseline overlay, hourly table below |
| **Models** | Leaderboard table (RMSE, MAE, R² on validation and test, % improvement over naive), model comparison chart, SHAP/LIME segmented toggle with top-10 feature bars and local waterfall |
| **EDA** | 2-column grid of chart cards: distributions, diurnal heatmap, correlation matrix, seasonal decomposition, imputation audit table |
| **Agents and drift** | Sub-tabs: Call log, Failed calls and fallbacks, Drift log, Graph trace. Filter row (agent, status, date) above a table |
| **Technical report** | Single centered column (800px), sticky table of contents on desktop, headings `h2`/`h3`, tables scroll horizontally on mobile |

---

## 8. Motion

| Use | Duration | Easing |
|---|---|---|
| Hover, press, focus | 120ms | `cubic-bezier(0.2, 0, 0, 1)` |
| Popovers, menus, tabs | 200ms | `cubic-bezier(0.2, 0, 0, 1)` |
| Theme switch | Instant (`disableTransitionOnChange`) to avoid a flash of mixed colors | none |
| First-load forecast line draw | 600ms, once | `ease-out` |

Rules:

- No entrance animations on cards or sections. The one load moment is the forecast line draw.
- Motion that responds to a click (tabs, menus, expanding log rows) is fine.
- Respect `prefers-reduced-motion: reduce` by turning off the line draw and all transitions longer than 0ms.

---

## 9. Accessibility

- Text contrast at least 4.5:1, large text and UI components at least 3:1, checked in **both** themes.
- Every AQI value shows its level name and icon next to the color. Every alert has a severity label and icon.
- Keyboard: all tabs, toggles, and table sorts reachable and operable; visible focus ring.
- Touch targets at least 44×44px on mobile.
- Charts: `aria-label`, plus a table view toggle.
- Tooltips open on focus as well as hover.
- Test every tab in light, dark, and system mode, and at 200% zoom.

---

## 10. Copy and labels

- Sentence case for all headings, buttons, labels, and tabs.
- Use user language: "Air quality", "Feels like", "Chance of rain", "Model accuracy", not internal names.
- Buttons say what happens: "Refresh data", "Show as table", "Download report".
- Empty states say what to do next: "No alerts right now. Alerts appear here when a forecast crosses a hazard threshold."
- Error states say what failed and what to try: "Couldn't load forecasts. Check your connection and refresh."
- Alert titles are plain: "Heatwave warning", "Unhealthy air quality", "Strong wind expected".
- Timestamps show Pakistan time (PKT) with a tooltip for UTC.

---

## 11. Implementation

### 11.1 Fonts and theme provider (`web/index.html` & `web/src/main.tsx`)

Load `Barlow Semi Condensed` and `Public Sans` from Google Fonts in `web/index.html`, along with a small anti-FOUC script matching the `theme` key used by `ThemeProvider`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>SkyCast (AgentWeather)</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600;700&family=Public+Sans:wght@400;500;600&display=swap"
      rel="stylesheet"
    />
    <script>
      (() => {
        const stored = localStorage.getItem("theme");
        const dark =
          stored === "dark" ||
          ((!stored || stored === "system") &&
            window.matchMedia("(prefers-color-scheme: dark)").matches);
        document.documentElement.classList.add(dark ? "dark" : "light");
      })();
    </script>
  </head>
  <body class="bg-background text-foreground font-sans antialiased">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

Then wrap `<App />` in shadcn's Vite `ThemeProvider` inside `web/src/main.tsx`:

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import { ThemeProvider } from "@/components/theme-provider.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider defaultTheme="system" disableTransitionOnChange>
      <App />
    </ThemeProvider>
  </StrictMode>
);
```

### 11.2 Theme toggle (`web/src/components/theme-toggle.tsx`)

```tsx
import { useTheme } from "@/components/theme-provider";
import { Moon, Sun, Monitor } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export function ThemeToggle() {
  const { setTheme } = useTheme();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Change theme">
          <Sun className="size-5 scale-100 dark:scale-0 transition-none" strokeWidth={1.75} />
          <Moon className="absolute size-5 scale-0 dark:scale-100 transition-none" strokeWidth={1.75} />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => setTheme("light")}><Sun className="size-4" /> Light</DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("dark")}><Moon className="size-4" /> Dark</DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("system")}><Monitor className="size-4" /> System</DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
```

### 11.3 AQI helper (`web/src/lib/aqi.ts`)

Tailwind only generates classes it can find as complete strings, so map each level to full class names instead of building them dynamically.

```ts
export const AQI_LEVELS = [
  { key: "good", max: 50, label: "Good",
    badge: "bg-aqi-good-bg text-aqi-good-fg border-aqi-good", fill: "var(--aqi-good)" },
  { key: "moderate", max: 100, label: "Moderate",
    badge: "bg-aqi-moderate-bg text-aqi-moderate-fg border-aqi-moderate", fill: "var(--aqi-moderate)" },
  { key: "sensitive", max: 150, label: "Unhealthy for sensitive groups",
    badge: "bg-aqi-sensitive-bg text-aqi-sensitive-fg border-aqi-sensitive", fill: "var(--aqi-sensitive)" },
  { key: "unhealthy", max: 200, label: "Unhealthy",
    badge: "bg-aqi-unhealthy-bg text-aqi-unhealthy-fg border-aqi-unhealthy", fill: "var(--aqi-unhealthy)" },
  { key: "very", max: 300, label: "Very unhealthy",
    badge: "bg-aqi-very-bg text-aqi-very-fg border-aqi-very", fill: "var(--aqi-very)" },
  { key: "hazardous", max: Infinity, label: "Hazardous",
    badge: "bg-aqi-hazardous-bg text-aqi-hazardous-fg border-aqi-hazardous", fill: "var(--aqi-hazardous)" },
] as const;

export const aqiLevel = (value: number) =>
  AQI_LEVELS.find((l) => value <= l.max) ?? AQI_LEVELS[AQI_LEVELS.length - 1];
```

### 11.4 Recharts usage

```tsx
{/* band is a [lower_bound, upper_bound] tuple */}
<Line dataKey="temperature" stroke="var(--chart-1)" strokeWidth={2} dot={false} />
<Line dataKey="forecast" stroke="var(--chart-1)" strokeWidth={2} strokeDasharray="6 4" dot={false} />
<Area dataKey="band" stroke="none" fill="var(--chart-1)" fillOpacity={0.15} />
<ReferenceLine x={nowTimestamp} stroke="var(--foreground)" strokeOpacity={0.4} label="Now" />
<CartesianGrid vertical={false} stroke="var(--border)" />
<XAxis tickLine={false} axisLine={false} tick={{ fill: "var(--muted-foreground)", fontSize: 12 }} />
```

### 11.5 `web/src/index.css` (single source of truth for all tokens)

Written for Tailwind v4 with shadcn/ui. If you are on Tailwind v3, the color values need to be converted to HSL channels.

```css
@import "tailwindcss";
@import "tw-animate-css";
@import "shadcn/tailwind.css";

@custom-variant dark (&:is(.dark *));

:root {
  /* Core UI */
  --background: #F6F8FB;
  --foreground: #0F172A;
  --card: #FFFFFF;
  --card-foreground: #0F172A;
  --popover: #FFFFFF;
  --popover-foreground: #0F172A;
  --primary: #0369A1;
  --primary-foreground: #FFFFFF;
  --secondary: #E8EEF6;
  --secondary-foreground: #0F172A;
  --muted: #EEF2F7;
  --muted-foreground: #64748B;
  --accent: #E0F2FE;
  --accent-foreground: #075985;
  --destructive: #B91C1C;
  --border: #E2E8F0;
  --input: #E2E8F0;
  --ring: #0EA5E9;

  --sidebar: #FFFFFF;
  --sidebar-foreground: #0F172A;
  --sidebar-primary: #0369A1;
  --sidebar-primary-foreground: #FFFFFF;
  --sidebar-accent: #E0F2FE;
  --sidebar-accent-foreground: #075985;
  --sidebar-border: #E2E8F0;
  --sidebar-ring: #0EA5E9;

  /* Metric chart colors */
  --chart-1: #EA580C; /* temperature */
  --chart-2: #7C3AED; /* aqi */
  --chart-3: #2563EB; /* precipitation */
  --chart-4: #0D9488; /* humidity */
  --chart-5: #CA8A04; /* uv */
  --chart-6: #64748B; /* wind */

  /* AQI: fill / text / tint */
  --aqi-good: #059669;       --aqi-good-fg: #065F46;       --aqi-good-bg: #D1FAE5;
  --aqi-moderate: #CA8A04;   --aqi-moderate-fg: #854D0E;   --aqi-moderate-bg: #FEF9C3;
  --aqi-sensitive: #EA580C;  --aqi-sensitive-fg: #9A3412;  --aqi-sensitive-bg: #FFEDD5;
  --aqi-unhealthy: #DC2626;  --aqi-unhealthy-fg: #991B1B;  --aqi-unhealthy-bg: #FEE2E2;
  --aqi-very: #9333EA;       --aqi-very-fg: #6B21A8;       --aqi-very-bg: #F3E8FF;
  --aqi-hazardous: #9F1239;  --aqi-hazardous-fg: #881337;  --aqi-hazardous-bg: #FFE4E6;

  /* Alerts */
  --alert-info: #0369A1;     --alert-info-bg: #E0F2FE;
  --alert-warning: #B45309;  --alert-warning-bg: #FEF3C7;
  --alert-critical: #B91C1C; --alert-critical-bg: #FEE2E2;

  /* Models */
  --model-sarimax: #7C3AED;
  --model-rf: #16A34A;
  --model-ridge: #D97706;
  --model-lstm: #E11D48;
  --model-naive: #64748B;

  /* Explainability and heatmaps */
  --shap-pos: #EA580C;
  --shap-neg: #0369A1;
  --corr-neutral: #F8FAFC;
  --heat-1: #F0F9FF; --heat-2: #BAE6FD; --heat-3: #38BDF8; --heat-4: #0284C7; --heat-5: #0C4A6E;

  /* Elevation */
  --shadow-card: 0 1px 2px rgb(15 23 42 / 0.04);
  --shadow-popover: 0 8px 24px rgb(15 23 42 / 0.12);
  --shadow-dialog: 0 16px 48px rgb(15 23 42 / 0.18);

  /* Layout */
  --sidebar-w: 248px;
  --sidebar-w-rail: 72px;
  --header-h: 64px;
  --bottom-nav-h: 64px;
  --page-max: 1440px;
}

.dark {
  --background: #0B1220;
  --foreground: #E6EDF7;
  --card: #111A2E;
  --card-foreground: #E6EDF7;
  --popover: #111A2E;
  --popover-foreground: #E6EDF7;
  --primary: #38BDF8;
  --primary-foreground: #0B1220;
  --secondary: #17233D;
  --secondary-foreground: #E6EDF7;
  --muted: #142038;
  --muted-foreground: #8FA0BC;
  --accent: #123349;
  --accent-foreground: #7DD3FC;
  --destructive: #F87171;
  --border: #1E2A44;
  --input: #1E2A44;
  --ring: #38BDF8;

  --sidebar: #0D1526;
  --sidebar-foreground: #E6EDF7;
  --sidebar-primary: #38BDF8;
  --sidebar-primary-foreground: #0B1220;
  --sidebar-accent: #123349;
  --sidebar-accent-foreground: #7DD3FC;
  --sidebar-border: #1E2A44;
  --sidebar-ring: #38BDF8;

  --chart-1: #FB923C;
  --chart-2: #A78BFA;
  --chart-3: #60A5FA;
  --chart-4: #2DD4BF;
  --chart-5: #FACC15;
  --chart-6: #94A3B8;

  --aqi-good: #34D399;       --aqi-good-fg: #34D399;       --aqi-good-bg: rgba(52,211,153,.15);
  --aqi-moderate: #FACC15;   --aqi-moderate-fg: #FACC15;   --aqi-moderate-bg: rgba(250,204,21,.15);
  --aqi-sensitive: #FB923C;  --aqi-sensitive-fg: #FB923C;  --aqi-sensitive-bg: rgba(251,146,60,.15);
  --aqi-unhealthy: #F87171;  --aqi-unhealthy-fg: #F87171;  --aqi-unhealthy-bg: rgba(248,113,113,.15);
  --aqi-very: #C084FC;       --aqi-very-fg: #C084FC;       --aqi-very-bg: rgba(192,132,252,.15);
  --aqi-hazardous: #E11D48;  --aqi-hazardous-fg: #FDA4AF;  --aqi-hazardous-bg: rgba(225,29,72,.22);

  --alert-info: #38BDF8;     --alert-info-bg: rgba(56,189,248,.12);
  --alert-warning: #FBBF24;  --alert-warning-bg: rgba(251,191,36,.12);
  --alert-critical: #F87171; --alert-critical-bg: rgba(248,113,113,.12);

  --model-sarimax: #A78BFA;
  --model-rf: #4ADE80;
  --model-ridge: #FBBF24;
  --model-lstm: #FB7185;
  --model-naive: #94A3B8;

  --shap-pos: #FB923C;
  --shap-neg: #38BDF8;
  --corr-neutral: #1E2A44;
  --heat-1: #0F1B30; --heat-2: #0C4A6E; --heat-3: #0284C7; --heat-4: #38BDF8; --heat-5: #E0F2FE;

  --shadow-card: none;
  --shadow-popover: 0 8px 24px rgb(0 0 0 / 0.5);
  --shadow-dialog: 0 16px 48px rgb(0 0 0 / 0.6);
}

@theme inline {
  /* Fonts */
  --font-sans: "Public Sans", ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif;
  --font-display: "Barlow Semi Condensed", ui-sans-serif, system-ui, sans-serif;

  /* Type scale */
  --text-hero: 4.5rem;     --text-hero--line-height: 1;
  --text-display: 3rem;    --text-display--line-height: 1.0833;
  --text-stat: 2rem;       --text-stat--line-height: 1.125;
  --text-h1: 1.875rem;     --text-h1--line-height: 1.2;
  --text-h2: 1.375rem;     --text-h2--line-height: 1.2727;
  --text-bulletin: 1.0625rem; --text-bulletin--line-height: 1.647;

  /* Radius hierarchy */
  --radius-sm: 6px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --radius-xl: 16px;
  --radius-2xl: 20px;

  /* Shadows */
  --shadow-card: var(--shadow-card);
  --shadow-popover: var(--shadow-popover);
  --shadow-dialog: var(--shadow-dialog);

  /* Core colors */
  --color-background: var(--background);
  --color-foreground: var(--foreground);
  --color-card: var(--card);
  --color-card-foreground: var(--card-foreground);
  --color-popover: var(--popover);
  --color-popover-foreground: var(--popover-foreground);
  --color-primary: var(--primary);
  --color-primary-foreground: var(--primary-foreground);
  --color-secondary: var(--secondary);
  --color-secondary-foreground: var(--secondary-foreground);
  --color-muted: var(--muted);
  --color-muted-foreground: var(--muted-foreground);
  --color-accent: var(--accent);
  --color-accent-foreground: var(--accent-foreground);
  --color-destructive: var(--destructive);
  --color-border: var(--border);
  --color-input: var(--input);
  --color-ring: var(--ring);
  --color-sidebar: var(--sidebar);
  --color-sidebar-foreground: var(--sidebar-foreground);
  --color-sidebar-primary: var(--sidebar-primary);
  --color-sidebar-primary-foreground: var(--sidebar-primary-foreground);
  --color-sidebar-accent: var(--sidebar-accent);
  --color-sidebar-accent-foreground: var(--sidebar-accent-foreground);
  --color-sidebar-border: var(--sidebar-border);
  --color-sidebar-ring: var(--sidebar-ring);

  /* Chart colors */
  --color-chart-1: var(--chart-1);
  --color-chart-2: var(--chart-2);
  --color-chart-3: var(--chart-3);
  --color-chart-4: var(--chart-4);
  --color-chart-5: var(--chart-5);
  --color-chart-6: var(--chart-6);

  /* AQI (enables bg-aqi-good-bg, text-aqi-good-fg, border-aqi-good, ...) */
  --color-aqi-good: var(--aqi-good);
  --color-aqi-good-fg: var(--aqi-good-fg);
  --color-aqi-good-bg: var(--aqi-good-bg);
  --color-aqi-moderate: var(--aqi-moderate);
  --color-aqi-moderate-fg: var(--aqi-moderate-fg);
  --color-aqi-moderate-bg: var(--aqi-moderate-bg);
  --color-aqi-sensitive: var(--aqi-sensitive);
  --color-aqi-sensitive-fg: var(--aqi-sensitive-fg);
  --color-aqi-sensitive-bg: var(--aqi-sensitive-bg);
  --color-aqi-unhealthy: var(--aqi-unhealthy);
  --color-aqi-unhealthy-fg: var(--aqi-unhealthy-fg);
  --color-aqi-unhealthy-bg: var(--aqi-unhealthy-bg);
  --color-aqi-very: var(--aqi-very);
  --color-aqi-very-fg: var(--aqi-very-fg);
  --color-aqi-very-bg: var(--aqi-very-bg);
  --color-aqi-hazardous: var(--aqi-hazardous);
  --color-aqi-hazardous-fg: var(--aqi-hazardous-fg);
  --color-aqi-hazardous-bg: var(--aqi-hazardous-bg);

  /* Alerts */
  --color-alert-info: var(--alert-info);
  --color-alert-info-bg: var(--alert-info-bg);
  --color-alert-warning: var(--alert-warning);
  --color-alert-warning-bg: var(--alert-warning-bg);
  --color-alert-critical: var(--alert-critical);
  --color-alert-critical-bg: var(--alert-critical-bg);

  /* Models and explainability */
  --color-model-sarimax: var(--model-sarimax);
  --color-model-rf: var(--model-rf);
  --color-model-ridge: var(--model-ridge);
  --color-model-lstm: var(--model-lstm);
  --color-model-naive: var(--model-naive);
  --color-shap-pos: var(--shap-pos);
  --color-shap-neg: var(--shap-neg);
}

@layer base {
  * { border-color: var(--border); }
  body { font-variant-numeric: tabular-nums; }
  :focus-visible { outline: 2px solid var(--ring); outline-offset: 2px; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0ms !important;
    transition-duration: 0ms !important;
  }
}
```

---

## 12. Build checklist

- [ ] No hex values in components or chart code, only `var(--token)` or Tailwind token classes
- [ ] Theme toggle works (Light, Dark, System) with no flash on reload
- [ ] Every AQI value shows color, level name, and icon
- [ ] Every chart has `aria-label` and a table view
- [ ] All six tabs checked in light and dark mode: tooltips, grid lines, SHAP/LIME bars, heatmaps
- [ ] Layout checked at 375, 768, 1024, 1280, and 1536px widths
- [ ] Bottom nav clears the phone safe area
- [ ] Keyboard-only pass: every control reachable with a visible focus ring
- [ ] Reduced-motion setting disables the forecast line draw
- [ ] `npm run build` passes with zero console errors
