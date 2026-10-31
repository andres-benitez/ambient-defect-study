"""Shared page head and CSS for the HTML reports."""

HEAD = '''<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600;700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: a long-form technical brief read top to bottom; numbered steps because the argument is a
   sequence; aluminium-grey neutrals with a slight blue bias; two series colours (cast blue, machined orange). */
:root{--bg:#f3f4f6;--surface:#fbfbfc;--ink:#14181d;--ink-2:#4b5560;--muted:#7d8793;--grid:#e0e4e9;--rule:#c6ccd3;
  --cast:#2a78d6;--mach:#eb6834;--dir:#1baf7a;--warn:#b4540a;--warn-bg:#fdf1e7;--note-bg:#eef3f9;--note:#2a5a8f;--good-bg:#edf7f1;
  --font-display:"Barlow Semi Condensed","Arial Narrow",Arial,sans-serif;--font-body:"Source Sans 3","Segoe UI",Helvetica,Arial,sans-serif;--font-mono:"JetBrains Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141719;--surface:#1c2024;--ink:#eef0f2;--ink-2:#bfc6cd;--muted:#8e97a0;--grid:#2b3136;--rule:#3a4147;
  --cast:#3987e5;--mach:#d95926;--dir:#199e70;--warn:#f0a56b;--warn-bg:#2a2017;--note-bg:#1b2634;--note:#8db4e2;--good-bg:#17261e;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141719;--surface:#1c2024;--ink:#eef0f2;--ink-2:#bfc6cd;--muted:#8e97a0;--grid:#2b3136;--rule:#3a4147;
  --cast:#3987e5;--mach:#d95926;--dir:#199e70;--warn:#f0a56b;--warn-bg:#2a2017;--note-bg:#1b2634;--note:#8db4e2;--good-bg:#17261e;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--font-body);font-size:17px;line-height:1.55;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:0 56px}
.wrap{max-width:960px;margin:0 auto;display:grid;gap:44px}
h1,h2,h3{font-family:var(--font-display);text-wrap:balance;margin:0;line-height:1.12}
h1{font-size:clamp(34px,5vw,50px);font-weight:700;letter-spacing:-.01em}
h2{font-size:clamp(24px,3vw,30px);font-weight:600}
h3{font-size:19px;font-weight:600;margin-top:6px}
p{margin:0;max-width:70ch}
ul,ol{margin:0;padding-left:22px;max-width:70ch} li{margin:4px 0}
.eyebrow{font-family:var(--font-mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
header{padding-top:40px;display:grid;gap:14px}
header .lede{font-size:clamp(19px,2.3vw,22px);color:var(--ink-2);max-width:66ch}
.toc{background:var(--surface);border:1px solid var(--grid);padding:16px 20px;display:grid;gap:6px}
.toc a{color:var(--ink);text-decoration:none;border-bottom:1px solid var(--rule)} .toc a:hover{border-bottom-color:var(--ink)}
.toc ol{columns:2;column-gap:28px;padding-left:20px;max-width:none} .toc li{break-inside:avoid;font-size:15px}
@media (max-width:640px){.toc ol{columns:1}}
section{display:grid;gap:16px;scroll-margin-top:16px}
.step{display:grid;gap:4px}
.step .eyebrow{color:var(--mach)}
figure{margin:0;background:var(--surface);border:1px solid var(--grid);padding:10px;display:grid;gap:8px}
figure img{width:100%;height:auto;display:block}
figure.pix img{image-rendering:pixelated}
figcaption{font-size:14px;color:var(--ink-2);line-height:1.45}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--grid)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:15px}
caption{text-align:left;padding:10px 12px 0;font-size:14px;color:var(--ink-2)}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--grid);vertical-align:top}
th{font-family:var(--font-mono);font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:500}
td.num{font-family:var(--font-mono);font-size:14px}
.box{border-left:4px solid var(--note);background:var(--note-bg);padding:14px 18px;display:grid;gap:6px;max-width:none}
.box.warn{border-left-color:var(--warn);background:var(--warn-bg)}
.box.good{border-left-color:var(--dir);background:var(--good-bg)}
.box b{font-family:var(--font-display);font-size:17px}
.box p,.box li{font-size:16px;color:var(--ink-2)}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1px;background:var(--grid);border:1px solid var(--grid)}
.fact{background:var(--surface);padding:14px 16px;display:grid;gap:4px;min-width:0}
.fact b{font-family:var(--font-display);font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.fact span{color:var(--ink-2);font-size:14px}
.verdict td:nth-child(2){font-weight:600}
.v-yes{color:var(--dir)} .v-no{color:var(--warn)} .v-part{color:var(--note)}
.missing{color:var(--warn);font-style:italic}
dl{display:grid;grid-template-columns:max-content 1fr;gap:6px 18px;max-width:none} dt{font-weight:600} dd{margin:0;color:var(--ink-2)}
@media (max-width:640px){dl{grid-template-columns:1fr} dd{margin-bottom:6px} body{font-size:16px}}
footer{font-size:13px;color:var(--muted);border-top:1px solid var(--grid);padding-top:14px;display:grid;gap:4px}
code{font-family:var(--font-mono);font-size:.9em}
sup{font-size:.7em}
</style>'''
