const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "..", "doubao_work2api", "web", "app.js"), "utf8");

function extractFunction(name) {
  const start = source.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `missing function ${name}`);
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
      if (depth === 0) return source.slice(start, index + 1);
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

testFormatTime();
testAccountImportDocuments();
testAccountImportFile();
testMissingKeyAccounts();
testMissingBindingSaveGuard();
testFilterAndSelection();
console.log("test_web.js: all tests passed");
