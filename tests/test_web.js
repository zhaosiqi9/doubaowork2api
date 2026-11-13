const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "..", "doubao_work2api", "web", "app.js"), "utf8");

function extractFunction(name) {
  const start = source.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `missing function ${name}`);
  const prefixStart = source.slice(Math.max(0, start - 6), start) === "async " ? start - 6 : start;
  const bodyStart = source.indexOf("{", start);
  let depth = 0;
  let quote = null;
  let escaped = false;
  let templateDepth = 0;
  for (let index = bodyStart; index < source.length; index += 1) {
    const character = source[index];
    if (quote) {
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === quote) quote = null;
      continue;
    }
    if (character === '"' || character === "'" || character === "`") {
      quote = character;
      continue;
    }
    if (character === "{") depth += 1;
    if (character === "}") {
      depth -= 1;
      if (depth === 0) return source.slice(prefixStart, index + 1);
    }
  }
  throw new Error(`unterminated function ${name}`);
}

function makeContext(extra = {}) {
  const context = {
    console,
    Intl,
    Number,
    String,
    Promise,
    Set,
    Map,
    Error,
    JSON,
    state: { accounts: [], selected: new Set() },
    ...extra,
  };
  vm.createContext(context);
  return context;
}

function loadFunctions(names, context) {
  const code = names.map(extractFunction).join("\n");
  vm.runInContext(code, context);
  return context;
}

function testFormatTime() {
  const context = loadFunctions(["timestampDate", "formatTime"], makeContext());
  assert.equal(context.formatTime("2026-11-13T02:03:04+00:00"), "10:03:04");
  assert.equal(context.formatTime(1763000000), "10:13:20");
  assert.equal(context.formatTime(1763000000000), "10:13:20");
  assert.equal(context.formatTime("1763000000"), "10:13:20");
  assert.equal(context.formatTime(""), "—");
  assert.equal(context.formatTime(null), "—");
  assert.equal(context.formatTime("not-a-date"), "—");
  assert.doesNotThrow(() => context.formatTime("2026-11-13T02:03:04Z"));
}

function testAccountImportDocuments() {
  const context = loadFunctions(["accountImportDocuments"], makeContext());
  const one = { id: "one", cookies: { sessionid: "secret" } };
  const two = { id: "two" };
  assert.deepEqual(JSON.parse(JSON.stringify(context.accountImportDocuments(one))), [one]);
  assert.deepEqual(JSON.parse(JSON.stringify(context.accountImportDocuments([one, two]))), [one, two]);
  assert.deepEqual(JSON.parse(JSON.stringify(context.accountImportDocuments({ accounts: [one, two] }))), [one, two]);
  assert.deepEqual(JSON.parse(JSON.stringify(context.accountImportDocuments(JSON.stringify({ accounts: [one] })))), [one]);
  assert.throws(() => context.accountImportDocuments({ accounts: [] }), /没有提供账号/);
  assert.throws(() => context.accountImportDocuments({ accounts: "bad" }), /accounts 字段必须是账号数组/);
  assert.throws(() => context.accountImportDocuments(["bad"]), /账号 JSON 必须是账号对象/);
  assert.throws(() => context.accountImportDocuments("not json"), /账号 JSON 格式无效/);
}

function testAccountImportFile() {
  const context = loadFunctions(["parseAccountImportFile"], makeContext());
  const one = { id: "one" };
  assert.deepEqual(JSON.parse(JSON.stringify(context.parseAccountImportFile(JSON.stringify(one)))), [one]);
  assert.deepEqual(JSON.parse(JSON.stringify(context.parseAccountImportFile(JSON.stringify([one])))), [one]);
  assert.deepEqual(JSON.parse(JSON.stringify(context.parseAccountImportFile(JSON.stringify({ accounts: [one] })))), [one]);
  assert.throws(() => context.parseAccountImportFile("not json"), /不是有效的 JSON/);
  assert.throws(() => context.parseAccountImportFile("null"), /文件必须包含账号/);
}

function testMissingKeyAccounts() {
  const context = loadFunctions(["missingKeyAccounts"], makeContext({
    state: { accounts: [{ id: "live" }] },
  }));
  assert.deepEqual(JSON.parse(JSON.stringify(context.missingKeyAccounts({ accounts: ["live", "gone"] }))), ["gone"]);
  assert.deepEqual(JSON.parse(JSON.stringify(context.missingKeyAccounts({ accounts: [] }))), []);
  assert.deepEqual(JSON.parse(JSON.stringify(context.missingKeyAccounts(null))), []);
}

function testMissingBindingSaveGuard() {
  const context = loadFunctions(["missingKeyAccounts", "selectedAccountsForSave"], makeContext({
    state: { accounts: [{ id: "live" }] },
  }));
  const key = { accounts: ["gone", "live"] };
  assert.throws(
    () => context.selectedAccountsForSave(key, ["gone", "live"]),
    /已删除.*不会自动转为允许全部账号/,
  );
  assert.deepEqual(JSON.parse(JSON.stringify(context.selectedAccountsForSave(key, ["live"]))), ["live"]);
  assert.deepEqual(JSON.parse(JSON.stringify(context.selectedAccountsForSave(key, []))), []);
  assert.deepEqual(JSON.parse(JSON.stringify(context.selectedAccountsForSave(null, []))), []);
}

function testFilterAndSelection() {
  const elements = {
    "#accountSearch": { value: "" },
    "#accountStatus": { value: "" },
    "#accountGroup": { value: "" },
    "#accountErrorCode": { value: "" },
  };
  const context = loadFunctions(["escapeHtml", "updateAccountFilters", "filteredAccounts", "visibleAccounts", "renderSelection"], makeContext({
    state: {
      accounts: [
        { id: "one", name: "Alpha", group: "A", state: "ready", error_code: 1 },
        { id: "two", name: "Beta", group: "B", state: "cooling", error_code: 2 },
      ],
      selected: new Set(["one"]),
      accountPage: 0,
      pageSize: 1,
    },
    $: (selector) => elements[selector],
    $$: (selector) => selector === "[data-batch]" ? elements.batchButtons : [],
  }));
  elements["#accountGroup"] = { value: "", innerHTML: "" };
  elements["#accountErrorCode"] = { value: "", innerHTML: "" };
  elements["#groupSuggestions"] = { innerHTML: "" };
  elements["#selectionSummary"] = { textContent: "" };
  elements.batchButtons = [{ disabled: false }, { disabled: false }];
  context.updateAccountFilters();
  assert.match(elements["#accountGroup"].innerHTML, /value="v:A"/);
  assert.match(elements["#accountErrorCode"].innerHTML, /value="v:1"/);
  assert.deepEqual(JSON.parse(JSON.stringify(context.filteredAccounts().map((account) => account.id))), ["one", "two"]);
  elements["#accountGroup"].value = "v:B";
  assert.deepEqual(JSON.parse(JSON.stringify(context.filteredAccounts().map((account) => account.id))), ["two"]);
  elements["#accountGroup"].value = "";
  elements["#accountErrorCode"].value = "v:1";
  assert.deepEqual(JSON.parse(JSON.stringify(context.filteredAccounts().map((account) => account.id))), ["one"]);
  elements["#accountErrorCode"].value = "";
  elements["#accountSearch"].value = "beta";
  assert.deepEqual(JSON.parse(JSON.stringify(context.filteredAccounts().map((account) => account.id))), ["two"]);
  assert.deepEqual(JSON.parse(JSON.stringify(context.visibleAccounts().map((account) => account.id))), ["two"]);
  context.renderSelection();
  assert.match(elements["#selectionSummary"].textContent, /筛选结果 1 个.*已选 1 个（筛选外 1 个）/);
  assert.equal(elements.batchButtons.every((button) => button.disabled === false), true);
  elements["#accountSearch"].value = "alpha";
  context.renderSelection();
  assert.match(elements["#selectionSummary"].textContent, /筛选结果 1 个.*已选 1 个（筛选外 0 个）/);
  elements["#accountSearch"].value = "";
  context.state.accounts.push({ id: "empty", state: "ready", group: "", error_code: "" });
  elements["#accountGroup"].value = "v:";
  assert.deepEqual(JSON.parse(JSON.stringify(context.filteredAccounts().map((account) => account.id))), ["empty"]);
  elements["#accountGroup"].value = "";
  elements["#accountErrorCode"].value = "v:";
  assert.deepEqual(JSON.parse(JSON.stringify(context.filteredAccounts().map((account) => account.id))), ["empty"]);
  context.state.selected.clear();
  context.renderSelection();
  assert.equal(elements.batchButtons.every((button) => button.disabled === true), true);
}

function testChatImageValidation() {
  const context = loadFunctions(["chatImageMime", "chatImageStats", "validateChatImageFile", "formatFileSize"], makeContext({
    CHAT_IMAGE_TYPES: new Set(["image/png", "image/jpeg", "image/webp", "image/gif"]),
    CHAT_IMAGE_LIMITS: { count: 8, fileBytes: 10 * 1024 * 1024, totalBytes: 20 * 1024 * 1024 },
  }));
  const png = { name: "one.png", size: 1024, type: "image/png" };
  const jpeg = { name: "two.jpg", size: 2 * 1024 * 1024, type: "image/jpeg" };
  assert.deepEqual(JSON.parse(JSON.stringify(context.validateChatImageFile(png))), { name: "one.png", size: 1024, type: "image/png" });
  assert.equal(context.chatImageMime({ type: "IMAGE/WEBP" }), "image/webp");
  assert.throws(() => context.validateChatImageFile({ name: "x.svg", size: 1, type: "image/svg+xml" }), /仅支持 PNG、JPEG、WebP、GIF/);
  assert.throws(() => context.validateChatImageFile({ name: "big.png", size: 10 * 1024 * 1024 + 1, type: "image/png" }), /单张不能超过 10 MiB/);
  assert.throws(() => context.validateChatImageFile(png, Array.from({ length: 8 }, (_, index) => ({ size: index }))), /最多选择 8 张/);
  assert.throws(() => context.validateChatImageFile(jpeg, [{ size: 19 * 1024 * 1024 }]), /总大小不能超过 20 MiB/);
  assert.deepEqual(JSON.parse(JSON.stringify(context.chatImageStats([png, jpeg]))), { count: 2, bytes: 1024 + 2 * 1024 * 1024 });
  assert.equal(context.formatFileSize(1024), "1.0 KiB");
  assert.equal(context.formatFileSize(2 * 1024 * 1024), "2.00 MiB");
}

function testChatContentParts() {
  const context = loadFunctions(["chatContentParts"], makeContext());
  assert.equal(context.chatContentParts("hello", []), "hello");
  assert.equal(context.chatContentParts("  hello  ", []), "hello");
  assert.deepEqual(
    JSON.parse(JSON.stringify(context.chatContentParts("", [{ name: "one.png", dataUrl: "data:image/png;base64,AA" }]))),
    [{ type: "image_url", image_url: { url: "data:image/png;base64,AA" } }],
  );
  assert.deepEqual(
    JSON.parse(JSON.stringify(context.chatContentParts("describe", [
      { name: "one.png", dataUrl: "data:image/png;base64,AA" },
      { name: "two.jpg", dataUrl: "data:image/jpeg;base64,BB" },
    ]))),
    [
      { type: "text", text: "describe" },
      { type: "image_url", image_url: { url: "data:image/png;base64,AA" } },
      { type: "image_url", image_url: { url: "data:image/jpeg;base64,BB" } },
    ],
  );
  assert.throws(() => context.chatContentParts("hello", [{ name: "pending.png", dataUrl: "" }]), /尚未读取完成/);
}

function testChatContentSnapshot() {
  const context = loadFunctions(["chatContentParts"], makeContext());
  const state = {
    chatImages: [
      { name: "one.png", dataUrl: "data:image/png;base64,AA" },
      { name: "two.png", dataUrl: "data:image/png;base64,BB" },
    ],
  };
  const snapshot = state.chatImages.map((image) => ({ ...image }));
  const content = context.chatContentParts("inspect", snapshot);
  state.chatImages.splice(0);
  assert.equal(content.length, 3);
  assert.equal(content[1].image_url.url, "data:image/png;base64,AA");
  assert.equal(content[2].image_url.url, "data:image/png;base64,BB");
}

function testChatImageCapabilityLabel() {
  const context = loadFunctions(["chatImageCapabilityLabel"], makeContext());
  assert.equal(context.chatImageCapabilityLabel({ image_input: true }), "已验证识图");
  assert.equal(context.chatImageCapabilityLabel({ image_input: null }), "");
  assert.equal(context.chatImageCapabilityLabel({ image_input: false }), "");
  assert.equal(context.chatImageCapabilityLabel({}), "");
}

function testChatImageEpochInvalidatesReads() {
  const context = loadFunctions([
    "formatFileSize", "chatImageStats", "chatImageKey", "renderChatImages",
    "clearChatImages", "removeChatImage",
  ], makeContext({
    state: {
      chatImages: [{ id: "one" }, { id: "two" }],
      chatImageEpoch: 3,
      chatImageQueue: Promise.resolve(),
    },
    $: () => ({ value: "selected", replaceChildren() {}, append() {}, hidden: false, textContent: "", disabled: false }),
    document: {
      createElement: () => ({
        className: "",
        dataset: {},
        setAttribute() {},
        append() {},
      }),
    },
    CHAT_IMAGE_LIMITS: { count: 8, fileBytes: 10 * 1024 * 1024, totalBytes: 20 * 1024 * 1024 },
  }));
  assert.equal(context.chatImageKey({ name: "one.png", size: 4, lastModified: 5 }), "one.png\u00004\u00005");
  context.removeChatImage("one");
  assert.equal(context.state.chatImageEpoch, 3);
  assert.deepEqual(JSON.parse(JSON.stringify(context.state.chatImages.map((image) => image.id))), ["two"]);
  context.clearChatImages();
  assert.equal(context.state.chatImageEpoch, 4);
  assert.deepEqual(JSON.parse(JSON.stringify(context.state.chatImages)), []);
  context.removeChatImage("missing");
  assert.equal(context.state.chatImageEpoch, 4);
}

function makeChatImageContext({ state, readChatImage }) {
  return loadFunctions([
    "formatFileSize", "chatImageMime", "chatImageStats", "validateChatImageFile",
    "chatImageKey", "renderChatImages", "addChatImageFiles", "enqueueChatImageFiles",
    "removeChatImage",
  ], makeContext({
    state,
    readChatImage,
    showToast: () => {},
    CHAT_IMAGE_TYPES: new Set(["image/png", "image/jpeg", "image/webp", "image/gif"]),
    CHAT_IMAGE_LIMITS: { count: 8, fileBytes: 10 * 1024 * 1024, totalBytes: 20 * 1024 * 1024 },
    $: () => ({ value: "", replaceChildren() {}, append() {}, hidden: false, textContent: "", disabled: false }),
    document: {
      createElement: () => ({
        className: "",
        dataset: {},
        setAttribute() {},
        append() {},
      }),
    },
  }));
}

async function testChatImageQueueUsesEnqueueEpoch() {
  const gate = {};
  gate.promise = new Promise((resolve) => { gate.release = resolve; });
  let reads = 0;
  const state = {
    sessionEpoch: 7,
    chatImageEpoch: 4,
    chatImages: [],
    chatImageQueue: Promise.resolve(),
  };
  const context = makeChatImageContext({
    state,
    readChatImage: async () => {
      reads += 1;
      await gate.promise;
      return "data:image/png;base64,AA";
    },
  });
  const file = { name: "old.png", size: 10, type: "image/png", lastModified: 1 };
  context.enqueueChatImageFiles([file]);
  state.sessionEpoch += 1;
  gate.release();
  await context.state.chatImageQueue;
  assert.equal(reads, 0);
  assert.deepEqual(JSON.parse(JSON.stringify(state.chatImages)), []);
}

async function testChatImageRemoveDoesNotCancelOtherReads() {
  const gates = [];
  const state = {
    sessionEpoch: 1,
    chatImageEpoch: 0,
    chatImages: [],
    chatImageQueue: Promise.resolve(),
  };
  const context = makeChatImageContext({
    state,
    readChatImage: async (file) => {
      const gate = {};
      gate.promise = new Promise((resolve) => { gate.release = resolve; });
      gates.push({ name: file.name, gate });
      await gate.promise;
      return `data:image/png;base64,${file.name}`;
    },
  });
  const first = { name: "first.png", size: 10, type: "image/png", lastModified: 1 };
  const second = { name: "second.png", size: 10, type: "image/png", lastModified: 2 };
  context.enqueueChatImageFiles([first, second]);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(state.chatImages.length, 1);
  assert.equal(state.chatImages[0].name, "first.png");
  context.removeChatImage(state.chatImages[0].id);
  assert.equal(state.chatImageEpoch, 0);
  gates[0].gate.release();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(state.chatImages.length, 1);
  assert.equal(state.chatImages[0].name, "second.png");
  gates[1].gate.release();
  await context.state.chatImageQueue;
  assert.equal(state.chatImages.length, 1);
  assert.equal(state.chatImages[0].name, "second.png");
  assert.equal(state.chatImages[0].pending, false);
  assert.equal(state.chatImages[0].dataUrl, "data:image/png;base64,second.png");
}

testFormatTime();
testAccountImportDocuments();
testAccountImportFile();
testMissingKeyAccounts();
testMissingBindingSaveGuard();
testFilterAndSelection();
testChatImageValidation();
testChatContentParts();
testChatContentSnapshot();
testChatImageCapabilityLabel();
testChatImageEpochInvalidatesReads();

(async () => {
  await testChatImageQueueUsesEnqueueEpoch();
  await testChatImageRemoveDoesNotCancelOtherReads();
  console.log("test_web.js: all tests passed");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
