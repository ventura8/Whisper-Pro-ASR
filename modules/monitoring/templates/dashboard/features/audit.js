function renderAuditDetails(item) {
    const id = _auditItemId(item);
    const toggle = _auditToggleIds(id);
    const caller = _auditCaller(item);
    const reqJson = _auditRequestJson(item);
    const resJson = _auditResponseJson(item);
    const openState = _auditOpenState(toggle);
    const userAgentDisplay = _auditUserAgentDisplay(caller.user_agent);

    return `
        <div style="margin-top:12px; border-top:1px solid var(--border-color); padding-top:12px;">
            <details class="js-toggle" data-toggle-id="${toggle.audit}" ${openState.audit}>
                <summary style="font-size:13px; color:var(--md-sys-color-secondary); margin-bottom:8px;">
                    <span class="material-icons-sharp" style="font-size:14px">policy</span> Audit & Caller Info
                </summary>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px; padding-left:16px;">
                    <div class="item-secondary"><span class="material-icons-sharp" style="font-size:12px">language</span> IP: ${escapeHtml(caller.ip || 'Local')}</div>
                    <div class="item-secondary" title="${userAgentDisplay.title}"><span class="material-icons-sharp" style="font-size:12px">devices</span> UA: ${userAgentDisplay.value}</div>
                </div>
            </details>
            <details class="js-toggle" data-toggle-id="${toggle.req}" ${openState.req}>
                <summary><span class="material-icons-sharp">code</span> Request Payload</summary>
                <div class="json-buffer">${reqJson}</div>
            </details>
            <details class="js-toggle" data-toggle-id="${toggle.res}" ${openState.res} style="margin-top:8px;">
                <summary><span class="material-icons-sharp">data_object</span> Response Payload</summary>
                <div class="json-buffer">${resJson}</div>
            </details>
        </div>
    `;
}

function _auditItemId(item) {
    return item.task_id ? item.task_id : item.filename;
}

function _auditToggleIds(id) {
    const raw = String(id ?? '');
    const base = raw.replace(/[^a-zA-Z0-9_.-]/g, '_');
    // Sanitizing to a safe DOM-id charset can collapse distinct ids (e.g. "a/b" and
    // "a:b" both become "a_b"), which would make their toggle states alias each
    // other. The suffix below is a lossless per-character encoding of the raw
    // (unsanitized) id -- not a hash -- so any two distinct raw ids always produce
    // distinct suffixes, with no collision risk regardless of string length/content.
    const suffix = raw ? `_${Array.from(raw).map((c) => c.codePointAt(0).toString(36)).join('.')}` : '';
    return {
        audit: `${base}${suffix}_audit`,
        req: `${base}${suffix}_req`,
        res: `${base}${suffix}_res`
    };
}

function _auditCaller(item) {
    return item.caller_info ? item.caller_info : {};
}

function _auditLooksLikeMediaPath(value) {
    const clean = String(value).trim().replace(/^["']|["']$/g, '');
    if (!clean.startsWith('/')) {
        return false;
    }
    const lower = clean.toLowerCase();
    return ['.mkv', '.mp4', '.avi', '.wav', '.m4a', '.mp3', '.flac', '.ts', '.mov', '.webm', '.wmv', '.mpg', '.mpeg']
        .some((ext) => lower.endsWith(ext));
}

function _auditExtractMediaPathKey(entries, existingLocalPath) {
    let localPath = existingLocalPath;
    const normalized = {};
    for (const [key, value] of entries) {
        if (_auditLooksLikeMediaPath(key)) {
            localPath = localPath || key.trim().replace(/^["']|["']$/g, '');
            continue;
        }
        normalized[key] = value;
    }
    return { normalized, localPath };
}

function _auditNormalizeRequestJson(requestPayload) {
    if (!requestPayload || typeof requestPayload !== 'object') {
        return {};
    }
    const { normalized, localPath } = _auditExtractMediaPathKey(
        Object.entries(requestPayload),
        requestPayload.local_path || null
    );
    if (localPath) {
        normalized.local_path = localPath;
    }
    return normalized;
}

function _auditRequestJson(item) {
    const requestPayload = item.request_json ? item.request_json : {};
    return escapeHtml(JSON.stringify(_auditNormalizeRequestJson(requestPayload), null, 2));
}

function _auditResponseJson(item) {
    const responsePayload = item.result ? item.result : (item.response_json ? item.response_json : {});
    return escapeHtml(JSON.stringify(responsePayload, null, 2));
}

function _auditOpenState(toggle) {
    return {
        audit: expandedElements.has(toggle.audit) ? 'open' : '',
        req: expandedElements.has(toggle.req) ? 'open' : '',
        res: expandedElements.has(toggle.res) ? 'open' : ''
    };
}

function _auditUserAgentDisplay(userAgent) {
    if (!userAgent) {
        return {
            title: escapeHtml(''),
            value: 'Not provided by client'
        };
    }
    const shortUa = userAgent.length > 30 ? userAgent.substring(0, 30) + '...' : userAgent;
    return {
        title: escapeHtml(userAgent),
        value: escapeHtml(shortUa)
    };
}