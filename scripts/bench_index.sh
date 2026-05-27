#!/usr/bin/env bash
# Aggregates benchmark reports from all 4 codebases into a single premium index.html.

set -euo pipefail

# Output index path
OUTPUT_DIR="target/criterion"
OUTPUT_FILE="$OUTPUT_DIR/index.html"
mkdir -p "$OUTPUT_DIR"

# Define the 4 codebases and their criterion target paths
declare -A CODEBASES
CODEBASES["sofar"]="/home/mukund-thiru/sofar/target/criterion"
CODEBASES["spill"]="/home/mukund-thiru/spill/target/criterion"
CODEBASES["star-randsrv"]="/home/mukund-thiru/star-randsrv/target/criterion"
CODEBASES["vyre"]="/media/mukund-thiru/SanthData/Santh/libs/performance/matching/vyre/target/criterion"

# Start building the premium HTML content
cat << 'EOF' > "$OUTPUT_FILE"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aggregated Benchmark Reports</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(255, 255, 255, 0.03);
            --card-border: rgba(255, 255, 255, 0.06);
            --primary-glow: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-green: #10b981;
        }
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            padding: 2rem 1rem;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        header {
            text-align: center;
            margin-bottom: 3.5rem;
            position: relative;
        }
        h1 {
            font-size: 2.75rem;
            font-weight: 700;
            background: var(--primary-glow);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -0.025em;
        }
        .subtitle {
            color: var(--text-muted);
            font-size: 1.1rem;
            font-weight: 300;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.75rem;
        }
        .codebase-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.75rem;
            backdrop-filter: blur(12px);
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.3s ease, box-shadow 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .codebase-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: var(--primary-glow);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .codebase-card:hover {
            transform: translateY(-4px);
            border-color: rgba(99, 102, 241, 0.4);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5), 0 0 1px rgba(99, 102, 241, 0.4);
        }
        .codebase-card:hover::before {
            opacity: 1;
        }
        .codebase-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .badge {
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(99, 102, 241, 0.15);
            color: #818cf8;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            border: 1px solid rgba(99, 102, 241, 0.3);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .bench-list {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }
        .bench-item {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: background 0.2s ease, border-color 0.2s ease;
        }
        .bench-item:hover {
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(255, 255, 255, 0.08);
        }
        .bench-link {
            color: var(--text-main);
            text-decoration: none;
            font-size: 0.95rem;
            font-weight: 400;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            width: 100%;
        }
        .bench-link::after {
            content: '→';
            margin-left: auto;
            color: var(--text-muted);
            transition: transform 0.2s ease, color 0.2s ease;
        }
        .bench-link:hover::after {
            transform: translateX(3px);
            color: #818cf8;
        }
        .no-benches {
            color: var(--text-muted);
            font-size: 0.9rem;
            font-style: italic;
        }
        footer {
            margin-top: 5rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            font-weight: 300;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 1.5rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Criterion Benchmark Reports</h1>
            <p class="subtitle">Aggregated performance insights across workspaces</p>
        </header>
        <div class="grid">
EOF

# Process each codebase and inject dynamic HTML cards
for name in "sofar" "spill" "star-randsrv" "vyre"; do
    path="${CODEBASES[$name]}"
    echo "Processing benchmarks for $name..."
    
    # Generate the codebase card
    cat << EOF >> "$OUTPUT_FILE"
            <div class="codebase-card">
                <div class="codebase-title">
                    <span>$name</span>
                    <span class="badge">Crate</span>
                </div>
                <ul class="bench-list">
EOF
    
    benches_found=0
    if [[ -d "$path" ]]; then
        # Find directories inside that represent benchmarks (they have a 'report' subdirectory)
        while IFS= read -r dir; do
            [[ -z "$dir" ]] && continue
            bench_name=$(basename "$dir")
            # Skip standard criterion directories
            if [[ "$bench_name" == "report" || "$bench_name" == "custom" ]]; then
                continue
            fi
            
            # Check if index.html exists in report folder
            if [[ -f "$dir/report/index.html" ]]; then
                report_path="$dir/report/index.html"
                benches_found=$((benches_found + 1))
                
                cat << EOL >> "$OUTPUT_FILE"
                    <li class="bench-item">
                        <a class="bench-link" href="file://$report_path" target="_blank">$bench_name</a>
                    </li>
EOL
            fi
        done < <(find "$path" -maxdepth 1 -type d 2>/dev/null | sort)
    fi
    
    if [[ "$benches_found" -eq 0 ]]; then
        cat << EOF >> "$OUTPUT_FILE"
                    <li class="no-benches">No local benchmarks found. Run cargo bench to generate reports.</li>
EOF
    fi
    
    cat << EOF >> "$OUTPUT_FILE"
                </ul>
            </div>
EOF
done

# End the HTML structure
cat << 'EOF' >> "$OUTPUT_FILE"
        </div>
        <footer>
            <p>Aggregated automatically &bull; Design: Antigravity</p>
        </footer>
    </div>
</body>
</html>
EOF

echo "Benchmark index page generated successfully at file://$(pwd)/$OUTPUT_FILE"
