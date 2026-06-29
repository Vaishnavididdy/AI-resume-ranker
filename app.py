"""Flask app — Enhanced AI Resume Ranker"""
from __future__ import annotations
import os
import time
import uuid
import traceback
from flask import Flask, render_template, request, jsonify, send_file, abort

from data_loader import load_job_description, load_candidates, FileParserError
from jd_intelligence import parse_jd
from feature_engineering import extract_candidate_features
from ranker import rank_candidates
from submission import build_submission_df, write_submission_csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024


def _save_upload(file_storage, prefix: str) -> str:
    if not file_storage or not file_storage.filename:
        abort(400, f"Missing file: {prefix}")
    safe_name = f"{prefix}_{uuid.uuid4().hex}_{os.path.basename(file_storage.filename)}"
    path = os.path.join(UPLOAD_DIR, safe_name)
    file_storage.save(path)
    return path


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/rank", methods=["POST"])
def rank():
    t0 = time.time()
    try:
        # Step 1: Save uploaded files
        jd_path = _save_upload(request.files.get("job_description"), "jd")
        cand_path = _save_upload(request.files.get("candidates"), "cand")
        
        # Step 2: Load and parse files
        try:
            jd_raw = load_job_description(jd_path)
            print(f"[app] Job description loaded: {len(jd_raw.get('raw_text', ''))} chars")
        except FileParserError as e:
            return jsonify({"error": f"Job description parsing failed: {str(e)}"}), 400
        
        # Step 3: JD Intelligence
        jd_features = parse_jd(jd_raw)
        print(f"[app] JD features: {jd_features.get('title', 'Unknown')}")
        
        # Step 4: Load candidates
        try:
            candidates_raw = load_candidates(cand_path)
            print(f"[app] Loaded {len(candidates_raw)} candidates")
        except FileParserError as e:
            return jsonify({"error": f"Candidate file parsing failed: {str(e)}"}), 400
        
        if not candidates_raw:
            return jsonify({"error": "No candidates parsed from the candidates file."}), 400
        
        # Step 5: Extract candidate features
        candidate_features = []
        extraction_errors = 0
        for i, c in enumerate(candidates_raw):
            try:
                features = extract_candidate_features(c)
                candidate_features.append(features)
            except Exception as e:
                print(f"[app] Error extracting features for candidate {i}: {e}")
                extraction_errors += 1
        
        if not candidate_features:
            return jsonify({"error": "Failed to extract features from any candidate"}), 400
        
        print(f"[app] Extracted features for {len(candidate_features)} candidates ({extraction_errors} errors)")
        
        # Step 6: Hybrid ranking
        ranked = rank_candidates(jd_features, candidate_features, top_n=100)
        print(f"[app] Ranked {len(ranked)} candidates")
        
        # Step 7: Build submission
        df = build_submission_df(ranked, jd_features, target_count=100)
        
        out_name = f"submission_{uuid.uuid4().hex}.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        write_submission_csv(df, out_path)
        
        elapsed = round(time.time() - t0, 2)
        preview = df.to_dict(orient="records")[:100]
        
        return jsonify({
            "elapsed_seconds": elapsed,
            "total_candidates": len(candidates_raw),
            "processed_candidates": len(candidate_features),
            "jd_title": jd_features.get("title", ""),
            "jd_must_have": jd_features.get("must_have", []),
            "jd_key_skills": jd_features.get("key_skills", []),
            "rows": preview,
            "download_url": f"/download/{out_name}",
        })
    
    except Exception as e:
        app.logger.exception("ranking failed")
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc() if app.debug else None
        }), 500


@app.route("/download/<name>")
def download(name: str):
    safe = os.path.basename(name)
    path = os.path.join(OUTPUT_DIR, safe)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, as_attachment=True, download_name="submission.csv", mimetype="text/csv")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)