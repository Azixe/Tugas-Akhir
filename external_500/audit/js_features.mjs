// Compute the extension's own features (utils.js, unmodified) for a list of URLs.
// Usage: node js_features.mjs <extension dir> <urls.json> <out.jsonl>
import fs from 'node:fs';
import path from 'node:path';

const [ext, urlsPath, outPath] = process.argv.slice(2);
const src = fs.readFileSync(path.join(ext, 'utils.js'), 'utf8');
const api = new Function(src + '\nreturn { tfidf, structural, preprocessXgb };')();
const rf = JSON.parse(fs.readFileSync(path.join(ext, 'tfidf_data.json'), 'utf8'));
const xgb = JSON.parse(fs.readFileSync(path.join(ext, 'tfidf_data_xgb.json'), 'utf8'));
const prep = JSON.parse(fs.readFileSync(path.join(ext, 'xgb_preprocessing.json'), 'utf8'));
const urls = JSON.parse(fs.readFileSync(urlsPath, 'utf8'));

const sparse = v => v.map((x, i) => [i, x]).filter(([, x]) => x !== 0);
const out = fs.createWriteStream(outPath);
for (const url of urls) {
    // Same expressions as predict() in background.js
    const rawRf = [...api.tfidf(url.toLowerCase(), rf), ...api.structural(url)];
    const rawXgb = [...api.tfidf(url.toLowerCase(), xgb), ...api.structural(url)];
    out.write(JSON.stringify({
        struct: api.structural(url),
        rf: sparse(rawRf.slice(0, rawRf.length - 9)),
        xgb_in: api.preprocessXgb(rawXgb, prep),
    }) + '\n');
}
out.end(() => console.log('wrote', urls.length, 'rows'));
