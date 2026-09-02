#!/usr/bin/env python3
"""Build the compact Box3D Metal PDF guide set."""

from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    PageBreak,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "pdf"
PAGE_WIDTH, PAGE_HEIGHT = A4

NAVY = colors.HexColor("#0A1A2F")
TEAL = colors.HexColor("#14B8A6")
CYAN = colors.HexColor("#38BDF8")
INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#526078")
PALE = colors.HexColor("#EAF7F5")
BLUE_PALE = colors.HexColor("#EDF7FE")
RULE = colors.HexColor("#CBD5E1")
WHITE = colors.white


def register_fonts() -> tuple[str, str, str]:
    candidates = [
        (
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/SFNSBold.ttf",
            "/System/Library/Fonts/SFNSMono.ttf",
        ),
        (
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Menlo.ttc",
        ),
    ]
    for regular, bold, mono in candidates:
        if all(Path(path).exists() for path in (regular, bold, mono)):
            try:
                pdfmetrics.registerFont(TTFont("GuideRegular", regular))
                pdfmetrics.registerFont(TTFont("GuideBold", bold))
                pdfmetrics.registerFont(TTFont("GuideMono", mono))
                return "GuideRegular", "GuideBold", "GuideMono"
            except Exception:
                continue
    return "Helvetica", "Helvetica-Bold", "Courier"


REGULAR, BOLD, MONO = register_fonts()


def styles():
    base = getSampleStyleSheet()
    return {
        "cover_kicker": ParagraphStyle(
            "cover_kicker", parent=base["Normal"], fontName=BOLD, fontSize=10,
            leading=13, textColor=TEAL, spaceAfter=6, tracking=1.4,
        ),
        "cover_title": ParagraphStyle(
            "cover_title", parent=base["Title"], fontName=BOLD, fontSize=31,
            leading=34, textColor=WHITE, alignment=TA_LEFT, spaceAfter=12,
        ),
        "cover_deck": ParagraphStyle(
            "cover_deck", parent=base["Normal"], fontName=REGULAR, fontSize=12,
            leading=17, textColor=colors.HexColor("#D8E7F5"), spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName=BOLD, fontSize=20,
            leading=24, textColor=NAVY, spaceBefore=3, spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName=BOLD, fontSize=12,
            leading=15, textColor=NAVY, spaceBefore=10, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontName=REGULAR, fontSize=8.7,
            leading=12.3, textColor=INK, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small", parent=base["BodyText"], fontName=REGULAR, fontSize=7.6,
            leading=10.2, textColor=MUTED, spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["BodyText"], fontName=REGULAR, fontSize=8.5,
            leading=11.8, leftIndent=11, firstLineIndent=-6, bulletIndent=3,
            textColor=INK, spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "code", parent=base["Code"], fontName=MONO, fontSize=7.1,
            leading=9.6, leftIndent=8, rightIndent=8, borderPadding=7,
            backColor=colors.HexColor("#F4F7FA"), borderColor=RULE,
            borderWidth=0.4, spaceBefore=4, spaceAfter=8,
        ),
        "callout": ParagraphStyle(
            "callout", parent=base["BodyText"], fontName=REGULAR, fontSize=8.5,
            leading=12, leftIndent=9, rightIndent=9, borderPadding=8,
            backColor=PALE, borderColor=TEAL, borderWidth=0.6,
            textColor=INK, spaceBefore=5, spaceAfter=8,
        ),
        "table_header": ParagraphStyle(
            "table_header", parent=base["BodyText"], fontName=BOLD, fontSize=7.2,
            leading=9.2, textColor=WHITE,
        ),
        "table_cell": ParagraphStyle(
            "table_cell", parent=base["BodyText"], fontName=REGULAR, fontSize=7.0,
            leading=9.1, textColor=INK,
        ),
    }


S = styles()


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullets(items: Iterable[str]):
    return [p(f"- {item}", "bullet") for item in items]


def code(text: str) -> Paragraph:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return p(escaped.replace("\n", "<br/>"), "code")


def table(headers, rows, widths=None):
    data = [[p(str(cell), "table_header") for cell in headers]]
    for row in rows:
        data.append([p(str(cell), "table_cell") for cell in row])
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BLUE_PALE]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return result


def page_chrome(canvas, doc, short_title):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_HEIGHT - 13 * mm, PAGE_WIDTH, 13 * mm, fill=1, stroke=0)
    canvas.setFont(BOLD, 7.5)
    canvas.setFillColor(WHITE)
    canvas.drawString(18 * mm, PAGE_HEIGHT - 8.5 * mm, "BOX3D METAL")
    canvas.setFont(REGULAR, 7.2)
    canvas.setFillColor(colors.HexColor("#C9D8E8"))
    canvas.drawRightString(PAGE_WIDTH - 18 * mm, PAGE_HEIGHT - 8.5 * mm, short_title)
    canvas.setStrokeColor(RULE)
    canvas.line(18 * mm, 13 * mm, PAGE_WIDTH - 18 * mm, 13 * mm)
    canvas.setFont(REGULAR, 6.8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8.5 * mm, "Lulzx/box3d-metal | baseline 47d7f7c | 2026-09-02")
    canvas.drawRightString(PAGE_WIDTH - 18 * mm, 8.5 * mm, f"{doc.page}")
    canvas.restoreState()


def cover(canvas, doc, title, deck, label):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, PAGE_HEIGHT - 9 * mm, PAGE_WIDTH, 9 * mm, fill=1, stroke=0)
    canvas.setFillColor(CYAN)
    canvas.circle(PAGE_WIDTH - 28 * mm, PAGE_HEIGHT - 42 * mm, 12 * mm, fill=1, stroke=0)
    canvas.setStrokeColor(colors.HexColor("#22547A"))
    canvas.setLineWidth(0.6)
    for offset in range(5):
        y = 34 * mm + offset * 10 * mm
        canvas.line(18 * mm, y, PAGE_WIDTH - 18 * mm, y + 28 * mm)
    frame = Frame(22 * mm, 75 * mm, PAGE_WIDTH - 44 * mm, 100 * mm, showBoundary=0)
    story = [p(label.upper(), "cover_kicker"), p(title, "cover_title"), p(deck, "cover_deck")]
    frame.addFromList(story, canvas)
    canvas.setFont(MONO, 7.5)
    canvas.setFillColor(colors.HexColor("#A7BDD2"))
    canvas.drawString(22 * mm, 22 * mm, "APPLE SILICON / METAL COMPUTE / BOX3D OVERLAY")
    canvas.restoreState()


class GuideDoc(BaseDocTemplate):
    def __init__(self, path: Path, title: str, short_title: str, deck: str, label: str):
        super().__init__(
            str(path), pagesize=A4, title=title, author="Box3D Metal contributors",
            leftMargin=18 * mm, rightMargin=18 * mm, topMargin=20 * mm,
            bottomMargin=18 * mm, pageCompression=1,
        )
        cover_frame = Frame(0, 0, 1, 1, id="cover-frame", showBoundary=0)
        body_frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates([
            PageTemplate(
                id="cover", frames=[cover_frame],
                onPage=lambda c, d: cover(c, d, title, deck, label),
                autoNextPageTemplate="content",
            ),
            PageTemplate(
                id="content", frames=[body_frame],
                onPage=lambda c, d: page_chrome(c, d, short_title),
            ),
        ])


def build(path, title, short_title, deck, label, story):
    doc = GuideDoc(path, title, short_title, deck, label)
    doc.build([PageBreak()] + story)


def quickstart_story():
    return [
        p("Reconstruct and run", "h1"),
        p("The public repository is an overlay: it contains no Box3D checkout. The bootstrap script clones the canonical upstream revision, verifies the patch, applies it, and builds a Release configuration."),
        p("Requirements", "h2"),
        *bullets(["Apple Silicon Mac and macOS Metal", "Xcode command-line tools", "Git, CMake, and Ninja"]),
        code("git clone https://github.com/Lulzx/box3d-metal.git\ncd box3d-metal\n./scripts/bootstrap.sh ../box3d-metal-worktree"),
        p("Validation", "h2"),
        code("../box3d-metal-worktree/build/metal-release/bin/test MetalTest\n../box3d-metal-worktree/build/metal-release/bin/metal_demo"),
        p(f"For a disposable end-to-end reconstruction, run <font name='{MONO}'>./scripts/verify-clean.sh</font>.", "callout"),
        p("Enable per world", "h1"),
        code("b3WorldId world = b3CreateWorld(&worldDef);\nif (!b3World_EnableMetal(world, 32768)) {\n    /* CPU path remains usable. */\n}"),
        p("The body threshold is deliberately caller-selected. It is not a universal recommendation; measure your world with full-step timing."),
        p("Experimental residency stages", "h2"),
        code("b3World_SetMetalFinalization(world, true);\nb3World_SetMetalBroadPhase(world, true);"),
        p("Finalization computes body and awake-shape results on Metal. Broad-phase mode updates resident leaves, refits internal bounds, and compacts exact-order candidates; CPU topology, filtering, and contact creation remain."),
        p("Read route telemetry", "h2"),
        code("b3MetalProfile p = b3World_GetMetalProfile(world);\nprintf(\"%s contacts=%llu pairs=%llu pairFallbacks=%llu\\n\",\n       p.deviceName, p.contactDispatchCount,\n       p.pairDispatchCount, p.pairFallbackCount);"),
        p("Dispatch counters prove a Metal stage ran. Pair dispatch means tree traversal and compaction, not GPU tree ownership, filtering, narrow phase, CCD, sleeping, or events.", "callout"),
        p("Next references", "h2"),
        *bullets(["docs/compatibility.md - exact supported and CPU fallback surface", "docs/performance.md - measured M4 Pro crossovers", "docs/troubleshooting.md - initialization, routing, and performance diagnosis"]),
    ]


def architecture_story():
    phases = [
        ("1", "Integrate velocity", "Forces, damping, gyroscopic term, speed limits"),
        ("2", "Warm start", "Ordered overflow, then conflict-free graph colors"),
        ("3", "Solve", "Biased contact and supported-joint iterations"),
        ("4", "Integrate position", "Position and normalized quaternion update"),
        ("5", "Relax", "Unbiased constraint iterations"),
        ("6", "Restitution", "Eligible convex and mesh contacts"),
        ("7", "Finalize", "Optional body state and awake-shape AABBs"),
        ("8", "Refit + pairs", "Resident leaves, internal bounds, stable candidates"),
    ]
    return [
        p("One ordered command graph", "h1"),
        p("Supported constrained worlds encode every substep into one Metal command buffer, wait once, and read body state back once. This minimizes ownership changes on Apple unified memory."),
        table(["Phase", "Stage", "Work"], phases, [13 * mm, 39 * mm, 110 * mm]),
        Spacer(1, 6),
        p("Unconstrained worlds use a smaller fused path. Unsupported constrained worlds keep the CPU solver and may use only the independent Metal position stage."),
        p("Unified memory is not free synchronization", "callout"),
        p("Persistent shared buffers avoid PCIe copies, but command submission, cache coherence, residency, and CPU/GPU ownership transitions still determine crossover."),
        p("Data layout and ownership", "h1"),
        *bullets([
            "Body state and compact body properties use geometrically grown persistent buffers.",
            "CPU contact preparation writes directly into shared Metal constraint allocations.",
            "Distance and parallel joints use separate compact type-dense buffers.",
            "Only accumulated joint solver state is unpacked after the command buffer.",
            "C/Metal size and offset assertions guard every shared ABI.",
            "Scalar shader vectors avoid native float3 padding surprises.",
        ]),
        p("Graph colors", "h2"),
        p("Constraints inside one non-overflow color never write the same dynamic body, so one GPU thread can own one constraint without atomics. Colors remain ordered with buffer barriers."),
        p("Deterministic overflow", "h2"),
        p("Overflow constraints may share bodies. A single Metal thread walks them in upstream order. Mixed distance/parallel overflow uses an eight-byte type/index descriptor, preserving order with one launch per phase."),
        p("Experimental pair traversal", "h2"),
        p("Metal retains Box3D tree topology, updates enlarged leaves, refits parents by height, and compacts candidates in exact upstream order. Pair records carry resident query metadata. A stable 256-lane scan emits one 32-byte record per enlarged shape, avoiding a full-result rescan and second body/shape-list walk during proxy bookkeeping. Topology changes invalidate residency; CPU filters and contact creation remain unchanged."),
        p("Apple GPU implementation choices", "h1"),
        *bullets([
            "One command buffer amortizes submission; shared storage avoids redundant copies; device-derived group widths follow occupancy; conflict-free colors avoid atomics while body-sharing overflow stays serial.",
        ]),
    ]


def compatibility_story():
    supported = [
        ("Fused integration", "Velocity and position across all substeps"),
        ("Convex contacts", "Normal, friction, tangent velocity, twist, rolling, restitution"),
        ("Mesh contacts", "Scalar multi-manifold solve and restitution"),
        ("Distance joints", "Rigid, spring, limit, motor, static bodies"),
        ("Parallel joints", "Soft alignment, torque limiting, static bodies"),
        ("Overflow", "Serial deterministic contact and mixed supported-joint order"),
        ("Broad phase", "Resident leaf/refit plus exact-order raw candidates"),
    ]
    errors = [
        ("Convex friction", "4.77e-7 transform", "3.98e-6 velocity"),
        ("Mesh contacts", "4.77e-7", "3.60e-6"),
        ("Distance modes/overflow", "1.07e-6", "8.27e-5"),
        ("Mixed distance/parallel", "7.45e-9", "1.79e-7"),
        ("Mixed joint overflow", "1.82e-6", "3.29e-4"),
        ("Static-body joints", "4.47e-8", "-"),
    ]
    return [
        p("Compatibility contract", "h1"),
        p(f"The reference is unmodified Box3D commit <font name='{MONO}'>47d7f7cc7e091142c08d11dc7d2e493c5d34f536</font>. Metal is opt-in and tolerance-equivalent, not bit-identical across platforms."),
        table(["GPU-resident surface", "Modes"], supported, [45 * mm, 117 * mm]),
        p("Explicit CPU boundary", "h2"),
        *bullets([
            "Broad-phase topology, pair filtering, narrow phase, and manifolds",
            "Contact and joint preparation",
            "Events, islands, sleeping, and CCD",
            "Recording, queries, topology mutation, and public API calls",
            "Filter, motor, prismatic, revolute, spherical, weld, and wheel solving",
            "Any joint requesting reaction-threshold events",
        ]),
        p("Unsupported constraints keep the reference CPU solve for that step. The position-only Metal stage may still run if its threshold is met.", "callout"),
        p("Differential evidence", "h1"),
        p("Matching CPU and Metal worlds run for multiple steps/substeps. Transforms and linear/angular velocities are compared after each scenario. Values below are observed maxima, not universal bounds."),
        table(["Scenario", "Transform", "Velocity"], errors, [65 * mm, 48 * mm, 49 * mm]),
        p("Acceptance matrix", "h2"),
        p("Float and double full suites; CPU-only Release; far-world VF64 containment; AddressSanitizer and UndefinedBehaviorSanitizer; float/double warning-as-error builds; shared dylib demo; install audit; and clean-clone reconstruction."),
        p("At (+1e8, -1e8), all 2,048 mixed-shape GPU AABBs contained the same-world CPU oracle. One upload plus ten resident refits was observed across ten contact steps.", "callout"),
        p("<b>Failure semantics.</b> Initialization failure returns false and leaves the world usable. Unsupported work increments fallback counters. Disabling Metal releases resources without destroying the world.", "callout"),
    ]


def performance_story():
    rows = [
        ("Position primitive", "~131K", "1.278x", "524,288"),
        ("Fused 4-substep primitive", "~2K", "10.584x", "524,288"),
        ("Unconstrained whole world", "largest point", "1.146x", "524,288"),
        ("Convex contacts", "~262K", "1.098x", "262,144"),
        ("Mesh contacts", "~8K", "1.646x", "131,072"),
        ("Distance joints", "~131K", "1.158x", "524,288"),
        ("Parallel joints", "none stable", "0.974x", "1,048,576"),
        ("Experimental GPU finalization", "none", "0.79x", "524,288"),
        ("GPU shape finalization", "none", "0.84x", "524,288"),
        ("GPU tree traversal", "~32K", "1.068x", "524,288"),
    ]
    return [
        p("Measured platform", "h1"),
        table(
            ["Component", "Recorded value"],
            [
                ("Machine", "MacBook Pro Mac16,8"),
                ("SoC", "Apple M4 Pro - 12 CPU / 16 GPU cores"),
                ("Memory", "24 GB unified"),
                ("OS / Metal", "macOS 26.7 / Metal 4 compiler 32023.883"),
                ("Compiler", "AppleClang 21.0.0 Release, FP contraction off"),
                ("World settings", "8 workers, 4 substeps"),
            ],
            [45 * mm, 117 * mm],
        ),
        Spacer(1, 7),
        p("Whole-world timings include preparation, submission, synchronization, finalization, and readback. The fused primitive is explicitly labeled because it excludes full-world bookkeeping.", "callout"),
        p("Results", "h1"),
        table(["Workload", "Crossover", "Best/large result", "Bodies"], rows, [54 * mm, 31 * mm, 42 * mm, 35 * mm]),
        p("What the data says", "h2"),
        *bullets([
            "Command fusion is the largest demonstrated architectural win.",
            "Body finalization is 27% slower and shape AABBs are 17.9% slower at 524,288 because CPU result streams remain.",
            "The earlier CPU-prefix traversal was 6.4% faster at 524,288; resident refit and VF64 have correctness evidence but no accepted new timing.",
            "Mesh contacts cross earlier than convex-wide stacks; distance joints benefit only at very large scale.",
            "Parallel joints expand compatibility without justifying default GPU routing.",
        ]),
        p("No universal threshold", "h1"),
        p("GPU frequency, worker scheduling, topology, contact density, and CPU-resident stages move crossover. The API therefore requires a caller-selected awake-body threshold."),
        code("./scripts/run-benchmarks.sh ../box3d-metal-worktree"),
        p("Run each complete executable in at least three separate processes and compare medians. Do not use GPU kernel time alone as a whole-world claim.", "callout"),
        p("Next performance work", "h2"),
        *bullets([
            "Make resident shape bounds authoritative and replace flat CPU bookkeeping with selective synchronization.",
            "Retain state across steps and add joint types only with mode matrices and whole-world evidence.",
        ]),
    ]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    documents = [
        (
            OUTPUT / "box3d-metal-quickstart.pdf",
            "Box3D Metal Quick Start",
            "Quick Start",
            "Reconstruct the pinned Box3D checkout, apply the Metal overlay, validate it, and enable GPU routing per world.",
            "Operator guide",
            quickstart_story(),
        ),
        (
            OUTPUT / "box3d-metal-architecture.pdf",
            "Box3D Metal Architecture",
            "Architecture",
            "A compact guide to the Apple Silicon command graph, shared-memory ownership, constraint colors, and deterministic overflow.",
            "Technical guide",
            architecture_story(),
        ),
        (
            OUTPUT / "box3d-metal-compatibility-validation.pdf",
            "Box3D Metal Compatibility and Validation",
            "Compatibility + Validation",
            "The exact GPU-resident surface, explicit CPU boundary, differential evidence, and completed acceptance matrix.",
            "Evidence guide",
            compatibility_story(),
        ),
        (
            OUTPUT / "box3d-metal-m4-pro-performance.pdf",
            "Box3D Metal M4 Pro Performance",
            "M4 Pro Performance",
            "End-to-end crossover evidence, interpretation, and the limits of the recorded Apple M4 Pro benchmark results.",
            "Benchmark brief",
            performance_story(),
        ),
    ]
    for path, title, short, deck, label, story in documents:
        build(path, title, short, deck, label, story)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
