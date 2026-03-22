# US Stocks Subsector Builder

Build an Excel for US listed stocks, excluding ADRs (rule-based), with major-sector + AI small-tag classification.

## What it does
- Pulls symbol universe from Nasdaq Trader:
  - `nasdaqlisted.txt`
  - `otherlisted.txt`
- Removes:
  - test issues
  - ETFs
  - ADR/ADS names by keyword rules
- Merges SEC ticker map (`cik`)
- Pulls SEC submissions profile (`sic`, `sicDescription`)
- Applies rule-based major sector + subsector
- Optional AI stage for:
  - `ai_subsector` (fine-grained)
  - `ai_small_tags` (multi-tags, e.g. `茶飲|光通訊|以巴戰爭`)
  - `company_summary_zh_tw` (繁中公司簡介)
- Exports multi-sheet Excel:
  - `all_stocks`
  - `summary`
  - one sheet per subsector

## Setup
```powershell
cd E:\VScode\my_python_project\stock_fetcher
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\requirements.txt
```

## Run
```powershell
python .\build_us_stock_universe.py --user-agent "stock-universe-builder/1.0 your_email@example.com"
```

Output default:
- `E:\VScode\my_python_project\stock_fetcher\output\us_stocks_subsectors.xlsx`

## Fast test (limited SEC calls)
```powershell
python .\build_us_stock_universe.py --max-sec-profiles 300 --user-agent "stock-universe-builder/1.0 your_email@example.com"
```

## Safe crawling (slower, lower ban risk)
```powershell
python .\build_us_stock_universe.py `
  --user-agent "stock-universe-builder/1.0 your_email@example.com" `
  --request-sleep-sec 0.6 `
  --max-retries 6
```

## AI second-stage classification
Set key first (OpenAI):
```powershell
$env:OPENAI_API_KEY="your_api_key"
```

Run with AI:
```powershell
python .\build_us_stock_universe.py `
  --ai-enable `
  --ai-model gpt-4.1-mini `
  --ai-max-rows 1200 `
  --ai-sleep-sec 0.9 `
  --request-sleep-sec 0.6 `
  --max-retries 6 `
  --user-agent "stock-universe-builder/1.0 your_email@example.com"
```

If you use a non-OpenAI provider with OpenAI-compatible API (for example Kimi), set key + base URL:
```powershell
$env:KIMI_API_KEY="your_kimi_key"
$env:KIMI_BASE_URL="https://api.moonshot.cn/v1"
python .\build_us_stock_universe.py `
  --ai-enable `
  --ai-model kimi-k2-0711-preview `
  --ai-only-unclassified `
  --ai-max-rows 1200 `
  --ai-sleep-sec 0.9 `
  --request-sleep-sec 0.6 `
  --max-retries 6 `
  --user-agent "stock-universe-builder/1.0 your_email@example.com"
```

If your provider uses Anthropic Messages style (your config says `api: anthropic-messages`):
```powershell
$env:KIMI_API_KEY="your_kimi_key"
python .\build_us_stock_universe.py `
  --ai-enable `
  --ai-api-type anthropic-messages `
  --ai-base-url "https://api.kimi.com/coding" `
  --ai-model "kimi-for-coding" `
  --ai-api-key $env:KIMI_API_KEY `
  --ai-only-unclassified `
  --ai-max-rows 1200 `
  --ai-sleep-sec 0.9 `
  --request-sleep-sec 0.6 `
  --max-retries 6 `
  --user-agent "stock-universe-builder/1.0 your_email@example.com"
```

## AI-only enrich existing Excel + Traditional Chinese translation
This mode does not crawl websites again. It reads existing `us_stocks_subsectors.xlsx`,
fills AI fields, and translates taxonomy text to Traditional Chinese.

```powershell
python .\enrich_ai_translate_excel.py `
  --in "E:/VScode/my_python_project/stock_fetcher/output/us_stocks_subsectors.xlsx" `
  --out "E:/VScode/my_python_project/stock_fetcher/output/us_stocks_subsectors_ai_zh_tw.xlsx" `
  --ai-api-type anthropic-messages `
  --ai-base-url "https://api.kimi.com/coding" `
  --ai-model "kimi-for-coding" `
  --ai-api-key "$env:KIMI_API_KEY" `
  --ai-only-missing `
  --ai-max-rows 1800 `
  --ai-sleep-sec 0.9
```

## Step 1 + 2: Investable universe + Multi-theme sheets
After AI file is ready, run this post-process script:
- Step 1: build `investable_all` by excluding SPAC / warrants / preferred / fund-like securities.
- Step 2: build `theme_*` sheets from `ai_small_tags_zh_tw` (one stock can appear in multiple themes).

```powershell
python .\postprocess_investable_and_themes.py `
  --in "E:/VScode/my_python_project/stock_fetcher/output/us_stocks_subsectors_ai_zh_tw.xlsx" `
  --out "E:/VScode/my_python_project/stock_fetcher/output/us_stocks_investable_themes.xlsx" `
  --min-theme-count 12 `
  --max-theme-sheets 200
```

Notes:
- `--min-theme-count` controls noise. If themes are too many/noisy, increase it to `20` or `30`.
- `--max-theme-sheets` limits workbook size. Keep around `100~250` for Excel usability.

## Notes
- ADR exclusion is name-rule based. It catches most ADR/ADS but not 100%.
- If you need stricter ADR exclusion, add a paid fundamental source with explicit `isAdr` flag and merge it.
- For AI-based subsector labeling, use this output as stage-1 input and run LLM classification on `security_name + sic_description`.

## Streamlit web UI (search/filter + notes + GitHub sync)
This project now includes:
- `streamlit_app.py` (main app)
- `data_loader.py` (Excel parser + internal schema)
- `notes_store.py` (SQLite notes store)
- `github_sync.py` (backup/restore `notes_export.json` via GitHub API)

### Local run
```powershell
cd E:\VScode\my_python_project\stock_fetcher
pip install -r .\requirements.txt
streamlit run .\streamlit_app.py
```

Default data file:
- `E:/VScode/my_python_project/stock_fetcher/output/us_stocks_investable_themes_zh_tw_fast.xlsx`

Default notes DB:
- `E:/VScode/my_python_project/stock_fetcher/output/notes.db`

### Streamlit Cloud secrets
Set in Streamlit Cloud -> App -> Settings -> Secrets:

```toml
EDITOR_PASSWORD = "your_password"
GITHUB_TOKEN = "ghp_xxx"
GITHUB_REPO = "yourname/yourrepo"
GITHUB_BRANCH = "main"
GITHUB_NOTES_PATH = "data/notes_export.json"
KIMI_API_KEY = "sk-kimi-xxxx"
KIMI_BASE_URL = "https://api.kimi.com/coding/v1"
KIMI_MODEL = "kimi-for-coding"
```

Notes:
- Visitors are read-only by default.
- Enter `EDITOR_PASSWORD` in sidebar to enable add/edit/delete notes.
- Click sync buttons in app:
  - `同步到 GitHub`: upload local notes JSON backup
  - `從 GitHub 還原`: restore notes from GitHub backup
- In `時事事件筆記`, click `AI 套用到股票清單` to generate:
  - event theme labels (e.g. SpaceX概念股 / 腦機接口 / 地緣衝突供應鏈)
  - impacted stock list with impact direction/confidence/reason
- A new `時事主題` tab is provided for theme-centric browsing.

## Monthly one-click update (PowerShell)
Use:
- [monthly_update.ps1](E:/VScode/my_python_project/stock_fetcher/monthly_update.ps1)

Run full pipeline:
```powershell
cd E:\VScode\my_python_project\stock_fetcher
$env:KIMI_API_KEY="sk-xxxx"
.\monthly_update.ps1
```

Optional:
```powershell
# Skip AI stage (requires existing us_stocks_subsectors_ai_zh_tw.xlsx)
.\monthly_update.ps1 -SkipAI

# Auto commit & push output files
.\monthly_update.ps1 -PushToGit
```
