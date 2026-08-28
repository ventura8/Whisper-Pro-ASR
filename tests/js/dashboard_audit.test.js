const path = require("path");
const { loadScriptInContext } = require("./helpers");

describe("audit.js", () => {
  let context;

  beforeEach(() => {
    context = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/core/utils.js"), {
      Date,
      expandedElements: new Set(),
    });
    context = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/features/audit.js"), context);
  });

  describe("_auditItemId", () => {
    it("prefers task_id over filename", () => {
      expect(context._auditItemId({ task_id: "t1", filename: "f1" })).toBe("t1");
    });

    it("falls back to filename when task_id missing", () => {
      expect(context._auditItemId({ filename: "f1" })).toBe("f1");
    });
  });

  describe("_auditToggleIds", () => {
    it("sanitizes ids to a safe DOM-id charset", () => {
      const toggle = context._auditToggleIds("weird id/with:chars!");
      expect(toggle.audit).toMatch(/^weird_id_with_chars__[a-z0-9.]+_audit$/);
      expect(toggle.req).toMatch(/^weird_id_with_chars__[a-z0-9.]+_req$/);
      expect(toggle.res).toMatch(/^weird_id_with_chars__[a-z0-9.]+_res$/);
    });

    it("handles null/undefined id", () => {
      const toggle = context._auditToggleIds(undefined);
      expect(toggle.audit).toBe("_audit");
    });

    it("preserves falsy-but-valid identifiers such as 0 (distinct from undefined)", () => {
      const toggleZero = context._auditToggleIds(0);
      const toggleUndef = context._auditToggleIds(undefined);
      expect(toggleZero.audit).toMatch(/^0_/);
      expect(toggleZero.audit).not.toBe(toggleUndef.audit);
      expect(toggleZero.req).not.toBe(toggleUndef.req);
      expect(toggleZero.res).not.toBe(toggleUndef.res);
    });

    it("gives distinct ids to different raw ids that sanitize to the same safe characters", () => {
      const toggleA = context._auditToggleIds("a/b");
      const toggleB = context._auditToggleIds("a:b");
      expect(toggleA.audit).not.toBe(toggleB.audit);
      expect(toggleA.req).not.toBe(toggleB.req);
      expect(toggleA.res).not.toBe(toggleB.res);
    });

    it("gives distinct ids for raw ids that sanitize to the same empty base ('!' vs '!@')", () => {
      const toggleBang = context._auditToggleIds("!");
      const toggleBangAt = context._auditToggleIds("!@");
      expect(toggleBang.audit).not.toBe(toggleBangAt.audit);
      expect(toggleBang.req).not.toBe(toggleBangAt.req);
      expect(toggleBang.res).not.toBe(toggleBangAt.res);
    });

    it("gives distinct ids for astral characters sharing the same leading surrogate", () => {
      // U+1F600 (😀) and U+1F601 (😁) both encode to the high surrogate D83D, differing
      // only in their low surrogate. charCodeAt(0) on the codepoint-iterated array
      // element would read just the high surrogate and collide; codePointAt(0) reads
      // the full astral codepoint and must not.
      const toggleA = context._auditToggleIds("\u{1F600}");
      const toggleB = context._auditToggleIds("\u{1F601}");
      expect(toggleA.audit).not.toBe(toggleB.audit);
      expect(toggleA.req).not.toBe(toggleB.req);
      expect(toggleA.res).not.toBe(toggleB.res);

      // And their open/expanded state must not alias each other either.
      context.expandedElements.add(toggleA.audit);
      expect(context._auditOpenState(toggleA).audit).toBe("open");
      expect(context._auditOpenState(toggleB).audit).toBe("");
    });
  });

  describe("_auditCaller", () => {
    it("returns caller_info when present", () => {
      const caller = { ip: "1.2.3.4" };
      expect(context._auditCaller({ caller_info: caller })).toBe(caller);
    });

    it("defaults to {} when caller_info missing", () => {
      expect(context._auditCaller({})).toEqual({});
    });
  });

  describe("_auditLooksLikeMediaPath / _auditNormalizeRequestJson", () => {
    it("strips media-path-looking keys into local_path", () => {
      const normalized = context._auditNormalizeRequestJson({ "/media/movie.mkv": "" , task: "transcribe" });
      expect(normalized.local_path).toBe("/media/movie.mkv");
      expect(normalized.task).toBe("transcribe");
      expect(normalized["/media/movie.mkv"]).toBeUndefined();
    });

    it("preserves an existing local_path over a discovered media-path key", () => {
      const normalized = context._auditNormalizeRequestJson({
        local_path: "/already/set.wav",
        "/other/path.mp3": "",
      });
      expect(normalized.local_path).toBe("/already/set.wav");
    });

    it("handles quoted media paths", () => {
      const normalized = context._auditNormalizeRequestJson({ '"/media/movie.mp4"': "" });
      expect(normalized.local_path).toBe("/media/movie.mp4");
    });

    it("leaves non-media-path keys untouched", () => {
      expect(context._auditLooksLikeMediaPath("not-a-path")).toBe(false);
      expect(context._auditLooksLikeMediaPath("/media/movie.txt")).toBe(false);
      expect(context._auditLooksLikeMediaPath("/media/movie.mkv")).toBe(true);
    });
  });

  describe("_auditRequestJson / _auditResponseJson", () => {
    it("stringifies and escapes the request payload", () => {
      const json = context._auditRequestJson({ request_json: { task: "<script>x</script>" } });
      expect(json).toContain("&lt;script&gt;");
      expect(json).not.toContain("<script>");
    });

    it("defaults request payload to {} when missing", () => {
      expect(context._auditRequestJson({})).toBe("{}");
    });

    it("prefers result over response_json for the response payload", () => {
      const json = context._auditResponseJson({ result: { ok: true }, response_json: { ok: false } });
      expect(json).toContain("&quot;ok&quot;: true");
      expect(json).not.toContain("false");
    });

    it("falls back from result to response_json to {}", () => {
      expect(context._auditResponseJson({ response_json: { a: 1 } })).toContain("&quot;a&quot;: 1");
      expect(context._auditResponseJson({})).toBe("{}");
    });
  });

  describe("_auditOpenState", () => {
    it("reflects expandedElements membership", () => {
      const toggle = { audit: "a", req: "r", res: "s" };
      expect(context._auditOpenState(toggle)).toEqual({ audit: "", req: "", res: "" });

      context.expandedElements.add("a");
      expect(context._auditOpenState(toggle).audit).toBe("open");
      expect(context._auditOpenState(toggle).req).toBe("");
    });
  });

  describe("_auditUserAgentDisplay", () => {
    it("shows placeholder text when UA is empty", () => {
      const display = context._auditUserAgentDisplay(undefined);
      expect(display.value).toBe("Not provided by client");
      expect(display.title).toBe("");
    });

    it("passes short UA through unmodified", () => {
      const display = context._auditUserAgentDisplay("curl/8.0");
      expect(display.value).toBe("curl/8.0");
      expect(display.title).toBe("curl/8.0");
    });

    it("truncates UA longer than 30 chars and keeps full string in title", () => {
      const longUa = "Mozilla/5.0 (X11; Linux x86_64) SomeExtraLongSuffix";
      const display = context._auditUserAgentDisplay(longUa);
      expect(display.value).toBe(longUa.substring(0, 30) + "...");
      expect(display.title).toBe(longUa);
    });

    it("escapes an XSS-bearing user agent", () => {
      const display = context._auditUserAgentDisplay("<img src=x onerror=alert(1)>");
      expect(display.value).not.toContain("<img");
      expect(display.title).not.toContain("<img");
    });
  });

  describe("renderAuditDetails", () => {
    it("renders IP, UA, and JSON payloads for a representative item", () => {
      const html = context.renderAuditDetails({
        task_id: "abc123",
        caller_info: { ip: "10.0.0.1", user_agent: "TestAgent/1.0" },
        request_json: { task: "transcribe" },
        result: { text: "hello" },
      });

      expect(html).toContain("10.0.0.1");
      expect(html).toContain("TestAgent/1.0");
      expect(html).toContain("transcribe");
      expect(html).toContain("hello");
    });

    it("defaults IP display to Local when missing", () => {
      const html = context.renderAuditDetails({ filename: "f.wav" });
      expect(html).toContain("Local");
    });

    it("escapes XSS-bearing caller IP", () => {
      const html = context.renderAuditDetails({
        task_id: "xss-1",
        caller_info: { ip: '<script>alert(1)</script>' },
      });
      expect(html).not.toContain("<script>alert(1)</script>");
      expect(html).toContain("&lt;script&gt;");
    });

    it("reflects open state for previously expanded toggles", () => {
      const toggle = context._auditToggleIds("abc123");
      context.expandedElements.add(toggle.audit);
      const html = context.renderAuditDetails({ task_id: "abc123" });
      expect(html).toContain(`data-toggle-id="${toggle.audit}" open`);
    });
  });
});
