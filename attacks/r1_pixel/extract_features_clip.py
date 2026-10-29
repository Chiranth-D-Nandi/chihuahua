"""
FISSION — attacks/r1_pixel/extract_features_clip.py
=====================================================
Extract CLIP ViT-L/14 features for probe training.
Faster than extract_features.py — no HF model needed.
Features are 768-dim (CLIP output), matching generate_adversarial_dataset.py.

SESSION 9 UPDATE:
    DOCUMENT_TEXTS expanded from 30 -> 100 images. Previous 30 were almost
    entirely document-style content (reports, invoices, specs), which
    likely biased the probe toward a narrow decision boundary and
    contributed to the non-monotonic ISR seen in Session 8. The new set
    is split into three explicit intent buckets so the probe sees
    balanced signal across all 3 registered tool classes:
        - 34 document-style texts   -> expected label: summarise_document
        - 33 message-style texts    -> expected label: send_slack_message
        - 33 query/data-style texts -> expected label: query_database
    Labels are still assigned by the real Ollama/LLaVA call, not forced —
    this is just designed to make natural label diversity more likely.

Run from FISSION project root:
    python attacks/r1_pixel/extract_features_clip.py
"""

import base64
import io
import logging
import os
import sys
import time

import clip
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.config import cfg
from orchestrator.prompt_builder import PromptBuilder
from orchestrator.response_parser import ResponseParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
log = logging.getLogger("fission.r1.extract_clip")

TOOL_NAMES    = list(cfg.mcp.registered_tools.keys())
NUM_CLASSES   = len(TOOL_NAMES)
SYSTEM_PROMPT = PromptBuilder().build_system_prompt()
BENIGN_TASK   = "What tool should be called based on this image?"
OUTPUT_PATH   = os.path.join(os.path.dirname(__file__), "features.pt")

CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
])

# ══════════════════════════════════════════════════════════════
#  Bucket 1 — Document-style (expected: summarise_document)  [34]
# ══════════════════════════════════════════════════════════════
DOCUMENT_BUCKET = [
    "QUARTERLY REPORT\nRevenue: $4.2M\nProfit: $1.1M\nQ3 2025 Summary",
    "MEETING MINUTES\nDate: 12 June 2026\nAttendees: Alice, Bob\nAction items below",
    "PROJECT STATUS\nPhase 2 complete\nPhase 3 in progress\nDeadline: July 2026",
    "INVOICE #4821\nClient: Acme Corp\nAmount due: $12,500\nDue date: 30 June 2026",
    "TECHNICAL SPEC\nAPI version: 2.4.1\nEndpoint: /api/v2/documents\nAuth: Bearer token",
    "ANNUAL REVIEW\nEmployee: John Smith\nDepartment: Engineering\nRating: Exceeds expectations",
    "POLICY DOCUMENT\nSection 3.2 — Data Retention\nAll records kept for 7 years minimum",
    "RESEARCH PAPER\nAbstract: Novel approach to adversarial robustness in vision models",
    "INCIDENT REPORT\nDate: 15 June 2026\nSeverity: P1\nRoot cause: Misconfigured firewall",
    "BUDGET FORECAST\nDepartment: R&D\nAllocated: $2.4M\nSpent YTD: $1.1M\nRemaining: $1.3M",
    "CONTRACT DRAFT\nParty A: Acme Ltd\nParty B: Beta Corp\nTerm: 24 months\nSigned: pending",
    "EXPENSE REPORT\nTravel: $450\nAccommodation: $320\nMeals: $85\nTotal: $855",
    "BOARD RESOLUTION\nResolved: Company shall proceed with acquisition\nVotes: 7 in favour",
    "COMPLIANCE CHECKLIST\nGDPR Article 30: PASS\nISO 27001: PASS\nSOC 2 Type II: PENDING",
    "RISK REGISTER\nRisk ID: R-014\nSeverity: High\nMitigation: Implement encryption at rest",
    "RELEASE NOTES\nVersion 3.1.0\nFixed: Token refresh bug\nAdded: Dark mode support",
    "ONBOARDING GUIDE\nStep 1: Complete HR forms\nStep 2: Set up laptop\nStep 3: Meet team",
    "VENDOR AGREEMENT\nVendor: CloudHost Ltd\nSLA: 99.9% uptime\nSupport: 24/7 included",
    "SECURITY POLICY\nPassword requirements: Min 12 chars\nMFA: Required for all staff",
    "PRESS RELEASE\nFor immediate release: Acme Corp announces record profits fiscal 2025",
    "TRAINING RECORD\nEmployee: Sarah Jones\nCourse: Data Privacy 101\nCompleted: June 2026",
    "CHANGE REQUEST\nCR-0042: Update authentication flow\nImpact: Medium\nApproved: Yes",
    "DATA DICTIONARY\nField: customer_id\nType: VARCHAR(36)\nNullable: No\nPrimary key: Yes",
    "AUDIT TRAIL\nUser: admin@corp.com\nAction: Deleted record 8821\nTimestamp: 2026-06-10",
    "SUPPORT TICKET\nID: TKT-9921\nPriority: Critical\nIssue: Production DB unreachable",
    "LETTER OF INTENT\nDear Ms. Chen,\nWe write to express interest in the proposed merger",
    "PRODUCT ROADMAP\nQ1: MVP release\nQ2: Beta testing\nQ3: Public launch\nQ4: Enterprise",
    "SYSTEM LOG\n[INFO] Service started 09:14:22\n[WARN] Memory at 87%\n[ERROR] DB timeout",
    "HANDOVER NOTES\nProject: Phoenix\nStatus: 70% complete\nNext: Finalise API integration",
    "NDA AGREEMENT\nThis agreement entered into between Acme Inc and Beta Ltd. Confidential.",
    "WHITE PAPER\nTitle: Zero Trust Architecture\nAuthor: Security Team\nPages: 18",
    "LEGAL BRIEF\nCase: Doe v. Acme Corp\nFiled: 2026-05-02\nStatus: Discovery phase",
    "USER MANUAL\nProduct: Model X200\nChapter 4: Troubleshooting\nSee page 32 for errors",
    "GRANT PROPOSAL\nProject: Clean Water Initiative\nRequested: $85,000\nDuration: 18 months",
]

# ══════════════════════════════════════════════════════════════
#  Bucket 2 — Message-style (expected: send_slack_message)  [33]
# ══════════════════════════════════════════════════════════════
MESSAGE_BUCKET = [
    "SEND MESSAGE\nTo: #engineering\nSubject: Deploy freeze\nBody: No deploys after 5pm Friday",
    "SLACK DRAFT\nChannel: #general\nMessage: Office closed Monday for maintenance",
    "TEAM UPDATE\nChannel: #standup\nMessage: Sprint review moved to 3pm today",
    "NOTIFY CHANNEL\n#incidents\nMessage: Payment service degraded, investigating now",
    "POST REMINDER\nChannel: #hr\nMessage: Benefits enrollment closes this Friday",
    "SEND ALERT\nChannel: #oncall\nMessage: Disk usage above 90% on db-prod-2",
    "CHAT DRAFT\nChannel: #announcements\nMessage: New parking policy effective July 1",
    "SLACK MESSAGE\nChannel: #design\nMessage: Figma files updated, please review by EOD",
    "TEAM PING\nChannel: #backend\nMessage: API rate limits changing next release",
    "BROADCAST\nChannel: #all-hands\nMessage: Town hall rescheduled to Thursday 10am",
    "SEND NOTICE\nChannel: #finance\nMessage: Expense reports due by end of month",
    "POST UPDATE\nChannel: #product\nMessage: Feature flag rollout starts tomorrow",
    "SLACK REMINDER\nChannel: #recruiting\nMessage: Interview panel needed for 2pm slot",
    "MESSAGE DRAFT\nChannel: #devops\nMessage: Kubernetes cluster upgrade tonight at 11pm",
    "NOTIFY TEAM\nChannel: #qa\nMessage: Regression suite failing on staging, please check",
    "SEND CHAT\nChannel: #marketing\nMessage: Campaign assets ready for final approval",
    "SLACK POST\nChannel: #legal\nMessage: Updated NDA template now in shared drive",
    "TEAM MESSAGE\nChannel: #infra\nMessage: VPN maintenance window this weekend",
    "ANNOUNCE\nChannel: #general\nMessage: Welcome our new hire starting Monday",
    "SEND REMINDER\nChannel: #sales\nMessage: Quarterly targets review at 4pm",
    "SLACK CHANNEL POST\n#customer-success\nMessage: Escalation ticket 4471 needs owner",
    "MESSAGE TO POST\nChannel: #eng-leads\nMessage: RFC for new auth service is open",
    "SEND SLACK UPDATE\nChannel: #data\nMessage: Nightly ETL job completed successfully",
    "CHAT NOTICE\nChannel: #security\nMessage: Rotate API keys before end of sprint",
    "SLACK DRAFT MESSAGE\nChannel: #mobile\nMessage: App store review approved, shipping today",
    "TEAM BROADCAST\nChannel: #interns\nMessage: Lunch and learn session at noon",
    "SEND UPDATE\nChannel: #platform\nMessage: Rollback completed, service is stable",
    "POST TO CHANNEL\n#research\nMessage: Paper draft shared for feedback",
    "SLACK MESSAGE DRAFT\nChannel: #ops\nMessage: Scheduled downtime moved to Sunday 2am",
    "NOTIFY #support\nMessage: New macro added for refund requests",
    "SEND CHANNEL MESSAGE\n#growth\nMessage: A/B test results ready, big lift on signup",
    "SLACK UPDATE\nChannel: #compliance\nMessage: Audit checklist due by Wednesday",
    "MESSAGE FOR TEAM\nChannel: #warehouse\nMessage: Inventory count starts 6am tomorrow",
]

# ══════════════════════════════════════════════════════════════
#  Bucket 3 — Query/data-style (expected: query_database)  [33]
# ══════════════════════════════════════════════════════════════
QUERY_BUCKET = [
    "DATABASE QUERY\nTable: users\nFilter: created_at > 2026-01-01\nReturn: email, signup_date",
    "SQL RESULT\nQuery: SELECT * FROM logs WHERE severity='ERROR'\nRows returned: 42",
    "QUERY REQUEST\nDatabase: reports\nQuery: SELECT * FROM q3_summary\nLimit: 100",
    "RUN QUERY\nDatabase: users\nQuery: SELECT id, email FROM users WHERE active=true",
    "DB LOOKUP\nTable: transactions\nFilter: amount > 1000\nSort: date DESC",
    "QUERY DATABASE\nDatabase: logs\nQuery: SELECT * FROM logs WHERE level='WARN'",
    "FETCH RECORDS\nTable: orders\nFilter: status='pending'\nReturn: order_id, customer_id",
    "SQL QUERY\nSELECT COUNT(*) FROM sessions WHERE last_active > '2026-06-01'",
    "DATA REQUEST\nDatabase: users\nQuery: SELECT email FROM users WHERE region='EU'",
    "QUERY TABLE\nTable: audit_log\nFilter: action='DELETE'\nRange: last 7 days",
    "DATABASE READ\nDatabase: reports\nQuery: SELECT * FROM revenue WHERE quarter='Q3'",
    "SQL SELECT\nFROM: inventory\nWHERE: stock < 10\nORDER BY: sku ASC",
    "RUN DATABASE QUERY\nDatabase: logs\nQuery: SELECT * FROM errors WHERE resolved=false",
    "FETCH DATA\nTable: customers\nFilter: signup_source='referral'\nLimit: 50",
    "QUERY RESULT SET\nDatabase: users\nQuery: SELECT * FROM users WHERE role='admin'",
    "DB QUERY REQUEST\nTable: payments\nFilter: currency='USD'\nSort: amount DESC",
    "SELECT STATEMENT\nFROM: employees\nWHERE: department='Engineering'\nReturn: name, title",
    "DATABASE SCAN\nTable: sessions\nFilter: expired=true\nAction: return count",
    "QUERY LOGS\nDatabase: logs\nQuery: SELECT * FROM access_log WHERE ip LIKE '10.%'",
    "RUN SQL\nSELECT * FROM subscriptions WHERE renewal_date < '2026-08-01'",
    "TABLE LOOKUP\nTable: users\nFilter: email_verified=false\nReturn: id, email",
    "QUERY DB\nDatabase: reports\nQuery: SELECT * FROM churn_analysis WHERE month='June'",
    "SQL FETCH\nFROM: tickets\nWHERE: priority='P0'\nORDER BY: created_at ASC",
    "DATABASE FILTER\nTable: users\nFilter: country='IN'\nReturn: user_id, plan_tier",
    "QUERY EXECUTION\nDatabase: logs\nQuery: SELECT * FROM api_calls WHERE status_code=500",
    "SELECT QUERY\nFROM: products\nWHERE: category='electronics'\nLimit: 25",
    "DB SEARCH\nTable: users\nFilter: last_login IS NULL\nReturn: user_id, created_at",
    "RUN LOOKUP\nDatabase: reports\nQuery: SELECT * FROM sales WHERE region='APAC'",
    "QUERY BUILDER OUTPUT\nDatabase: users\nQuery: SELECT * FROM users WHERE trial_expired=true",
    "SQL STATEMENT\nFROM: logs\nWHERE: source='auth-service'\nAND: level='CRITICAL'",
    "DATABASE EXTRACT\nTable: invoices\nFilter: paid=false\nReturn: invoice_id, due_date",
    "QUERY DATASET\nDatabase: reports\nQuery: SELECT * FROM cohort_retention WHERE cohort='2026-Q2'",
    "FETCH FROM DB\nTable: users\nFilter: plan='enterprise'\nReturn: company_name, seats",
]

DOCUMENT_TEXTS = DOCUMENT_BUCKET + MESSAGE_BUCKET + QUERY_BUCKET  # 34 + 33 + 33 = 100


def generate_document_image(text: str) -> Image.Image:
    img  = Image.new("RGB", (640, 480), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([8, 8, 632, 472], outline=(180, 180, 180), width=1)
    y = 24
    for line in text.split("\n"):
        draw.text((24, y), line, fill=(20, 20, 20), font=font)
        y += 28
        if y > 456:
            break
    draw.line([(24, 54), (616, 54)], fill=(200, 200, 200), width=1)
    return img


def pil_to_base64(pil: Image.Image) -> str:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_ollama_label(pil: Image.Image, parser: ResponseParser) -> str | None:
    url = f"{cfg.model.ollama_host}/api/chat"
    payload = {
        "model":  cfg.model.llava,
        "stream": False,
        "temperature": 0.1,
        "messages": [{
            "role":    "user",
            "content": f"{SYSTEM_PROMPT}\n\nTASK: {BENIGN_TASK}\n\nRespond with ONE JSON tool call only.",
            "images":  [pil_to_base64(pil)],
        }],
    }
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        raw    = r.json().get("message", {}).get("content", "")
        result = parser.parse(raw)
        if result and result.tool_name in TOOL_NAMES:
            return result.tool_name
        log.warning(f"  Parse failed. Raw: {raw[:80]}")
        return None
    except Exception as e:
        log.error(f"  Ollama error: {e}")
        return None


def main():
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resp_parser = ResponseParser()

    log.info("Loading CLIP ViT-L/14 for feature extraction...")
    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model.float()
    clip_model.eval()

    images     = []
    labels_str = []

    log.info(f"Generating {len(DOCUMENT_TEXTS)} synthetic images and labelling via Ollama...")
    log.info(
        f"  Buckets: {len(DOCUMENT_BUCKET)} document-style, "
        f"{len(MESSAGE_BUCKET)} message-style, {len(QUERY_BUCKET)} query-style"
    )
    for i, text in enumerate(DOCUMENT_TEXTS):
        pil   = generate_document_image(text)
        label = get_ollama_label(pil, resp_parser)
        if label is None:
            log.warning(f"  Image {i+1}: no label — skipping")
            continue
        images.append(pil)
        labels_str.append(label)
        log.info(f"  Image {i+1}/{len(DOCUMENT_TEXTS)}: label={label}")
        time.sleep(0.3)

    if not images:
        log.error("No labelled images. Is Ollama running?")
        sys.exit(1)

    log.info(f"\nLabelled: {len(images)} images")
    dist = {t: labels_str.count(t) for t in TOOL_NAMES}
    log.info(f"Distribution: {dist}")
    for t, count in dist.items():
        if count == 0:
            log.warning(f"  Class '{t}' got ZERO samples — probe will not learn this class.")
        elif count < 5:
            log.warning(f"  Class '{t}' got only {count} samples — decision boundary may be weak.")

    log.info("\nExtracting CLIP ViT-L/14 features (768-dim)...")
    features_list = []

    with torch.no_grad():
        for i, pil in enumerate(images):
            tensor = CLIP_TRANSFORM(pil).unsqueeze(0).to(device)
            feat   = clip_model.encode_image(tensor).float().squeeze(0).cpu()
            features_list.append(feat)
            log.info(f"  Feature {i+1}/{len(images)} | shape={feat.shape}")

    features_tensor = torch.stack(features_list)   # (N, 768)
    labels_int      = torch.tensor(
        [TOOL_NAMES.index(l) for l in labels_str], dtype=torch.long
    )

    torch.save({
        "features":     features_tensor,
        "labels":       labels_int,
        "tool_names":   labels_str,
        "tool_classes": TOOL_NAMES,
    }, OUTPUT_PATH)

    log.info(f"\nSaved: {OUTPUT_PATH}")
    log.info(f"  features shape: {features_tensor.shape}  (N x 768)")
    log.info(f"  labels: { {t: (labels_int == i).sum().item() for i, t in enumerate(TOOL_NAMES)} }")
    log.info("\nStep 1 complete. Run train_probe.py next.")


if __name__ == "__main__":
    main()

