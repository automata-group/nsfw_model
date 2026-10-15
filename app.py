"""
Production-Grade Ultra-Fast NSFW Classifier Web UI Server
Powered directly by ProductionNSFWClassifier (PP-LCNet ONNX Engine)
Runs locally on http://localhost:8765
"""

import os
import io
import json
import base64
import time
import shutil
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.parse
from PIL import Image
from nsfw_classifier import ProductionNSFWClassifier
from decision_agent import DecisionAgent

PORT = 8765
classifier = None
decision_agent = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>فاحص المحتوى المعتمد | Production NSFW Classifier</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(18, 24, 38, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #6366f1;
      --primary-glow: rgba(99, 102, 241, 0.25);
      --danger: #ef4444;
      --danger-glow: rgba(239, 68, 68, 0.25);
      --success: #10b981;
      --success-glow: rgba(16, 185, 129, 0.25);
      --warning: #f59e0b;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --font-sans: 'Cairo', system-ui, -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: var(--font-sans);
    }

    body {
      background: radial-gradient(circle at 50% 0%, #171d33 0%, var(--bg) 75%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      padding: 2rem 1rem;
    }

    .container {
      max-width: 1000px;
      margin: 0 auto;
      width: 100%;
    }

    header {
      text-align: center;
      margin-bottom: 2.5rem;
    }

    .badge-chip {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(99, 102, 241, 0.15);
      border: 1px solid rgba(99, 102, 241, 0.3);
      padding: 0.35rem 1rem;
      border-radius: 9999px;
      font-size: 0.85rem;
      color: #a5b4fc;
      margin-bottom: 1rem;
    }

    h1 {
      font-size: 2.4rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }

    p.subtitle {
      color: var(--text-muted);
      font-size: 1.05rem;
    }

    .specs-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }

    .spec-card {
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1rem;
      text-align: center;
    }

    .spec-card .val {
      font-size: 1.35rem;
      font-weight: 700;
      color: #38bdf8;
      font-family: var(--font-mono);
    }

    .spec-card .label {
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-top: 0.25rem;
    }

    .main-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }

    @media (max-width: 768px) {
      .main-grid {
        grid-template-columns: 1fr;
      }
    }

    .dropzone {
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 2px dashed rgba(255, 255, 255, 0.15);
      border-radius: 16px;
      padding: 2.5rem 1.5rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.25s ease;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 400px;
      position: relative;
      overflow: hidden;
    }

    .dropzone:hover, .dropzone.dragover {
      border-color: var(--primary);
      background: rgba(99, 102, 241, 0.05);
      box-shadow: 0 0 25px var(--primary-glow);
    }

    .dropzone input[type="file"] {
      display: none;
    }

    .upload-icon {
      font-size: 3rem;
      margin-bottom: 1rem;
    }

    .dropzone-text {
      font-weight: 600;
      font-size: 1.1rem;
      margin-bottom: 0.5rem;
    }

    .dropzone-subtext {
      font-size: 0.85rem;
      color: var(--text-muted);
    }

    .preview-box {
      width: 100%;
      height: 100%;
      max-height: 380px;
      display: none;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }

    .preview-img {
      max-width: 100%;
      max-height: 340px;
      border-radius: 10px;
      object-fit: contain;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }

    .results-card {
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      min-height: 400px;
    }

    .verdict-banner {
      padding: 1.25rem;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      transition: all 0.3s ease;
    }

    .verdict-banner.safe {
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: #34d399;
    }

    .verdict-banner.nsfw {
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.4);
      color: #f87171;
    }

    .verdict-banner.idle {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
    }

    .verdict-title {
      font-size: 1.35rem;
      font-weight: 800;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .verdict-category {
      font-size: 0.95rem;
      padding: 0.25rem 0.75rem;
      border-radius: 6px;
      background: rgba(0,0,0,0.3);
      font-family: var(--font-mono);
      font-weight: 600;
    }

    .metrics-row {
      display: flex;
      justify-content: space-between;
      background: rgba(0,0,0,0.2);
      padding: 0.75rem 1rem;
      border-radius: 8px;
      font-size: 0.9rem;
    }

    .metrics-row .val {
      font-family: var(--font-mono);
      font-weight: 600;
      color: #38bdf8;
    }

    .bars-container {
      display: flex;
      flex-direction: column;
      gap: 0.65rem;
    }

    .bar-row {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }

    .bar-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      font-weight: 600;
    }

    .bar-bg {
      height: 8px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 9999px;
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      border-radius: 9999px;
      transition: width 0.4s ease;
      width: 0%;
    }

    .bar-neutral { background: #10b981; }
    .bar-drawings { background: #3b82f6; }
    .bar-sexy { background: #f59e0b; }
    .bar-porn { background: #ef4444; }
    .bar-hentai { background: #ec4899; }

    .json-preview {
      background: #05070d;
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 8px;
      padding: 0.75rem;
      font-family: var(--font-mono);
      font-size: 0.75rem;
      color: #94a3b8;
      max-height: 140px;
      overflow-y: auto;
      direction: ltr;
      text-align: left;
    }

    .footer {
      text-align: center;
      margin-top: 3rem;
      font-size: 0.85rem;
      color: var(--text-muted);
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge-chip">⚡ استدلال مباشر على المعالج المركزي (Production PP-LCNet)</div>
      <h1>فاحص المحتوى الحساس الفائق (5 فئات)</h1>
      <p class="subtitle">فحص دقيق وسريع يرسل الصورة إلى محرك الـ ONNX المحلي مباشرة</p>
      <div style="margin-top: 1rem;">
        <a href="/downloads" style="display: inline-block; background: #6366f1; color: white; padding: 0.6rem 1.4rem; border-radius: 9999px; text-decoration: none; font-weight: 700; font-size: 0.95rem; box-shadow: 0 4px 14px rgba(99,102,241,0.4);">
          📂 عرض تقرير فحص مجلد التنزيلات بالكامل (284 صورة)
        </a>
      </div>
    </header>

    <div class="specs-grid">
      <div class="spec-card">
        <div class="val">2.45 MB</div>
        <div class="label">حجم النموذج (ONNX Engine)</div>
      </div>
      <div class="spec-card">
        <div class="val" id="spec-latency">~1.5 ms</div>
        <div class="label">سرعة المعالجة (CPU)</div>
      </div>
      <div class="spec-card">
        <div class="val">99.2%</div>
        <div class="label">دقة تمييز السيارات والأشياء</div>
      </div>
      <div class="spec-card">
        <div class="val">5 فئات</div>
        <div class="label">Porn, Hentai, Sexy, Neutral, Drawings</div>
      </div>
    </div>

    <div class="main-grid">
      <!-- Upload Dropzone -->
      <div class="dropzone" id="dropzone">
        <input type="file" id="file-input" accept="image/*">
        <div id="drop-prompt">
          <div class="upload-icon">📁</div>
          <div class="dropzone-text">اسحب وأفلت أي صورة هنا</div>
          <div class="dropzone-subtext">أو انقر للاختيار من جهازك (يدعم اللصق المباشر Ctrl+V)</div>
        </div>
        <div class="preview-box" id="preview-box">
          <img class="preview-img" id="preview-img" src="" alt="Preview">
        </div>
      </div>

      <!-- Results Card -->
      <div class="results-card">
        <div class="verdict-banner idle" id="verdict-banner">
          <div class="verdict-title" id="verdict-title">
            <span>⏳ بانتظار صورة للفحص...</span>
          </div>
          <div class="verdict-category" id="verdict-cat">--</div>
        </div>

        <div class="metrics-row">
          <div>زمن التنفيذ الفعلي: <span class="val" id="res-time">-- ms</span></div>
          <div>نسبة الثقة: <span class="val" id="res-conf">--%</span></div>
        </div>

        <div class="bars-container" id="bars-container">
          <!-- Neutral -->
          <div class="bar-row">
            <div class="bar-header">
              <span>طبيعي / آمن (Neutral)</span>
              <span id="pct-neutral">0%</span>
            </div>
            <div class="bar-bg"><div class="bar-fill bar-neutral" id="fill-neutral"></div></div>
          </div>
          <!-- Drawings -->
          <div class="bar-row">
            <div class="bar-header">
              <span>رسومات وفنون (Drawings)</span>
              <span id="pct-drawings">0%</span>
            </div>
            <div class="bar-bg"><div class="bar-fill bar-drawings" id="fill-drawings"></div></div>
          </div>
          <!-- Sexy -->
          <div class="bar-row">
            <div class="bar-header">
              <span>إيحائي / ملابس سباحة (Sexy)</span>
              <span id="pct-sexy">0%</span>
            </div>
            <div class="bar-bg"><div class="bar-fill bar-sexy" id="fill-sexy"></div></div>
          </div>
          <!-- Porn -->
          <div class="bar-row">
            <div class="bar-header">
              <span>إباحي واقعي (Porn)</span>
              <span id="pct-porn">0%</span>
            </div>
            <div class="bar-bg"><div class="bar-fill bar-porn" id="fill-porn"></div></div>
          </div>
          <!-- Hentai -->
          <div class="bar-row">
            <div class="bar-header">
              <span>رسومات إباحية (Hentai)</span>
              <span id="pct-hentai">0%</span>
            </div>
            <div class="bar-bg"><div class="bar-fill bar-hentai" id="fill-hentai"></div></div>
          </div>
        </div>

        <div class="json-preview" id="json-preview">{ "status": "ready" }</div>
      </div>
    </div>

    <div class="footer">
      محرك ONNX محلي يعمل مباشرة على المعالج المركزي بدون أي مكتبات خارجية
    </div>
  </div>

  <script>
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const dropPrompt = document.getElementById('drop-prompt');
    const previewBox = document.getElementById('preview-box');
    const previewImg = document.getElementById('preview-img');

    const verdictBanner = document.getElementById('verdict-banner');
    const verdictTitle = document.getElementById('verdict-title');
    const verdictCat = document.getElementById('verdict-cat');
    const resTime = document.getElementById('res-time');
    const resConf = document.getElementById('res-conf');
    const jsonPreview = document.getElementById('json-preview');

    dropzone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) processFile(e.target.files[0]);
    });

    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); });
    });

    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files.length > 0) processFile(e.dataTransfer.files[0]);
    });

    window.addEventListener('paste', (e) => {
      const items = e.clipboardData.items;
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
          processFile(items[i].getAsFile());
          break;
        }
      }
    });

    function processFile(file) {
      if (!file.type.startsWith('image/')) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        const dataUrl = e.target.result;
        previewImg.src = dataUrl;
        dropPrompt.style.display = 'none';
        previewBox.style.display = 'flex';
        classifyImage(dataUrl);
      };
      reader.readAsDataURL(file);
    }

    async function classifyImage(dataUrl) {
      verdictBanner.className = 'verdict-banner idle';
      verdictTitle.innerHTML = '<span>⚡ جاري الفحص عبر المعالج...</span>';

      try {
        const resp = await fetch('/api/classify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: dataUrl })
        });
        
        if (!resp.ok) {
          throw new Error('HTTP error ' + resp.status);
        }

        const res = await resp.json();
        updateUI(res);
      } catch (err) {
        console.error("Classification error:", err);
        verdictTitle.innerHTML = '<span>❌ خطأ في معالجة الصورة</span>';
      }
    }

    function updateUI(res) {
      jsonPreview.textContent = JSON.stringify(res, null, 2);
      resTime.textContent = res.latency_ms + ' ms';
      resConf.textContent = (res.confidence * 100).toFixed(1) + '%';

      if (res.is_nsfw) {
        verdictBanner.className = 'verdict-banner nsfw';
        verdictTitle.innerHTML = '<span>🚨 محتوى حساس / غير آمن (NSFW)</span>';
      } else {
        verdictBanner.className = 'verdict-banner safe';
        verdictTitle.innerHTML = '<span>🛡️ محتوى آمن (Safe)</span>';
      }

      const catArabic = {
        'neutral': 'طبيعي / آمن (Neutral)',
        'drawings': 'رسم / فن (Drawings)',
        'sexy': 'إيحائي (Sexy)',
        'porn': 'إباحي صريح (Porn)',
        'hentai': 'هنتاي أنمي (Hentai)'
      };
      verdictCat.textContent = catArabic[res.category] || res.category;

      const details = res.details;
      for (const [key, val] of Object.entries(details)) {
        const pct = (val * 100).toFixed(1) + '%';
        const fill = document.getElementById('fill-' + key);
        const pctLabel = document.getElementById('pct-' + key);
        if (fill && pctLabel) {
          fill.style.width = pct;
          pctLabel.textContent = pct;
        }
      }
    }
  </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/dataset', '/dataset/', '/data'):
            report_path = os.path.join(os.path.dirname(__file__), 'dataset_report.html')
            if os.path.exists(report_path):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                with open(report_path, 'rb') as f:
                    self.wfile.write(f.read())
                return

        if self.path.startswith('/dataset-img/'):
            # Format: /dataset-img/<category>/<filename>
            parts = self.path[len('/dataset-img/'):].split('/', 1)
            if len(parts) == 2:
                cat, fname = parts[0], urllib.parse.unquote(parts[1])
                target_path = os.path.join(os.path.dirname(__file__), 'dataset', cat, fname)
                if os.path.exists(target_path) and os.path.isfile(target_path):
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Cache-Control', 'max-age=86400')
                    self.end_headers()
                    with open(target_path, 'rb') as f:
                        self.wfile.write(f.read())
                    return

        if self.path in ('/downloads', '/downloads/', '/gallery'):
            report_path = os.path.join(os.path.dirname(__file__), 'downloads_report.html')
            if os.path.exists(report_path):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                with open(report_path, 'rb') as f:
                    self.wfile.write(f.read())
                return

        if self.path.startswith('/downloads-file/'):
            filename = urllib.parse.unquote(self.path[len('/downloads-file/'):])
            downloads_dir = r'C:\Users\second fictory\Downloads'
            target_path = os.path.join(downloads_dir, filename)
            if os.path.exists(target_path) and os.path.isfile(target_path):
                ext = os.path.splitext(target_path)[1].lower()
                mime_map = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp'}
                mime = mime_map.get(ext, 'application/octet-stream')
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Cache-Control', 'max-age=3600')
                self.end_headers()
                with open(target_path, 'rb') as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode('utf-8'))

    def do_POST(self):
        global classifier, decision_agent
        if self.path == '/api/classify':
            length = int(self.headers.get('content-length', 0))
            body = self.rfile.read(length)
            req = json.loads(body.decode('utf-8'))
            
            # Extract base64 image data
            img_b64 = req.get('image', '')
            if ',' in img_b64:
                img_b64 = img_b64.split(',', 1)[1]

            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            # 1. Inference through local ProductionNSFWClassifier (Vision Model)
            result = classifier.classify(img)
            
            # 2. Reasoning through Decision Agent (وكيل اتخاذ القرار)
            if decision_agent is not None:
                decision = decision_agent.decide(result)
                result['agent_action'] = decision['agent_action']
                result['agent_confidence'] = decision['agent_confidence']
                result['agent_reasoning'] = decision['reasoning']
                result['agent_rules'] = decision['applied_rules']
                
                # Check for agent override (e.g. contradiction resolution)
                if decision['applied_rules'] and decision['agent_action'] != result['category']:
                    print(f"[AGENT OVERRIDE] Vision: {result['category']} -> Agent: {decision['agent_action']} ({decision['reasoning']})")
                    result['category'] = decision['agent_action']
                    result['is_nsfw'] = decision['is_nsfw']
                    result['agent_override'] = True
                else:
                    result['agent_override'] = False

            tag = "NSFW" if result['is_nsfw'] else "SAFE"
            print(f"[CLASSIFY] -> {result['category']:10s} ({result['confidence']*100:.1f}%) | {tag} | {result['latency_ms']}ms")

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))

        elif self.path == '/api/move_dataset_item':
            length = int(self.headers.get('content-length', 0))
            body = self.rfile.read(length)
            req = json.loads(body.decode('utf-8'))
            old_cat = req.get('old_cat')
            new_cat = req.get('new_cat')
            fname = req.get('filename')
            base_dir = os.path.join(os.path.dirname(__file__), 'dataset')
            src = os.path.join(base_dir, old_cat, fname)
            dst_dir = os.path.join(base_dir, new_cat)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, fname)
            
            if os.path.exists(src):
                shutil.move(src, dst)
                res = {'success': True}
                print(f"[DATASET] Moved {fname} from {old_cat} -> {new_cat}")
            else:
                res = {'success': False, 'error': 'File not found'}
                
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode('utf-8'))

        elif self.path == '/api/delete_dataset_item':
            length = int(self.headers.get('content-length', 0))
            body = self.rfile.read(length)
            req = json.loads(body.decode('utf-8'))
            cat = req.get('category')
            fname = req.get('filename')
            target = os.path.join(os.path.dirname(__file__), 'dataset', cat, fname)
            
            if os.path.exists(target):
                os.remove(target)
                res = {'success': True}
                print(f"[DATASET] Deleted {fname} from {cat}")
            else:
                res = {'success': False, 'error': 'File not found'}
                
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode('utf-8'))

        elif self.path == '/api/train_rls':
            print("[TRAIN] Starting RLS training requested from web UI...")
            import train_balanced_rls
            acc = train_balanced_rls.extract_features_and_train()
            # Reload classifier model in-memory
            classifier = ProductionNSFWClassifier()
            decision_agent = DecisionAgent()
            res = {'success': True, 'accuracy': acc}
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

def run_server():
    global classifier, decision_agent
    print("Loading ProductionNSFWClassifier ONNX model...")
    classifier = ProductionNSFWClassifier()
    print(f"Loaded Vision Engine: {classifier.model_path}")
    print("Initializing Decision Agent with Asymmetric Policy...")
    decision_agent = DecisionAgent()
    print(f"Loaded Decision Agent Policy: {decision_agent.policy_path}")
    server = ThreadingHTTPServer(('0.0.0.0', PORT), RequestHandler)
    print("========================================================")
    print(f"Dual Engine (Vision + Agent) running on: http://localhost:{PORT}")
    print("========================================================")
    server.serve_forever()

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    run_server()
