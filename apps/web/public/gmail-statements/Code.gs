// Paste this script into your own Apps Script project and run setup once.
const API_URL = "https://wealth-api.example.invalid/v1/statement-inbox/uploads";
const UPLOAD_KEY = "replace-this-with-your-Wealth-upload-key";
// Statements come from these domains (e.g. donotreply@camsonline.com). Gmail's from:
// matches whole words, so a bare "cams" would miss "camsonline.com".
const KNOWN_SENDERS = [
  "camsonline.com",
  "kfintech.com",
  "mfcentral.com",
  "nsdl.co.in",
  "nsdl.com",
  "cdslindia.com",
  "cdslstatement.com",
];
const IMPORTED_LABEL = "Wealth/Imported";
const RETRY_DELAY_MS = 50000;
// Daily runs look at recent mail. The first run looks back a year, but sends only the
// newest statement from each provider, so Inbox doesn't fill with old ones.
const DAILY_DAYS = 45;
const FIRST_RUN_DAYS = 365;

// Shown in the Apps Script execution log, so a run never looks like "nothing happened".
function _log(text) {
  if (typeof console !== "undefined") console.log(text);
}

function setup() {
  const label = GmailApp.getUserLabelByName(IMPORTED_LABEL);
  if (!label) GmailApp.createLabel(IMPORTED_LABEL);

  const properties = PropertiesService.getScriptProperties();
  properties.setProperties({ API_URL: API_URL, UPLOAD_KEY: UPLOAD_KEY }, true);

  ScriptApp.getProjectTriggers().forEach(function (trigger) {
    if (trigger.getHandlerFunction() === "run") {
      ScriptApp.deleteTrigger(trigger);
    }
  });
  ScriptApp.newTrigger("run").timeBased().everyDays(1).atHour(7).create();
  // Check once now, so statements already in Gmail don't wait for tomorrow morning.
  return run();
}

function _searchQuery(days) {
  const senderSearch = KNOWN_SENDERS.map(function (sender) {
    return "from:" + sender;
  }).join(" ");
  // Gmail search writes a nested label "Wealth/Imported" as "wealth-imported".
  const labelSearch = IMPORTED_LABEL.toLowerCase().replace(/[\/\s]+/g, "-");
  return (
    "has:attachment filename:pdf newer_than:" +
    (days || DAILY_DAYS) +
    "d -label:" +
    labelSearch +
    " {" +
    senderSearch +
    "}"
  );
}

function _senderOf(thread) {
  const from = String(thread.getMessages()[0].getFrom() || "").toLowerCase();
  for (let index = 0; index < KNOWN_SENDERS.length; index += 1) {
    if (from.indexOf(KNOWN_SENDERS[index]) !== -1) return KNOWN_SENDERS[index];
  }
  return from;
}

// Gmail returns the newest threads first.
function _newestPerSender(threads) {
  const seen = {};
  return threads.filter(function (thread) {
    const sender = _senderOf(thread);
    if (seen[sender]) return false;
    seen[sender] = true;
    return true;
  });
}

function _isPdf(attachment) {
  const contentType = String(attachment.getContentType() || "").toLowerCase();
  const filename = String(attachment.getName() || "").toLowerCase();
  return contentType === "application/pdf" || filename.endsWith(".pdf");
}

function _postPdf(message, attachment, apiUrl, uploadKey) {
  const file = attachment.setName(attachment.getName() || "statement.pdf");
  const payload = {
    file: file,
    sender: message.getFrom(),
    subject: message.getSubject() || "",
    received_at: message.getDate().toISOString(),
    message_id: message.getId(),
  };
  const options = {
    method: "post",
    headers: { Authorization: "StatementKey " + uploadKey },
    payload: payload,
    muteHttpExceptions: true,
  };

  for (let attempt = 0; attempt < 2; attempt += 1) {
    let response;
    try {
      response = UrlFetchApp.fetch(apiUrl, options);
    } catch (_error) {
      if (attempt === 0) {
        Utilities.sleep(RETRY_DELAY_MS);
        continue;
      }
      _log("Couldn't reach Wealth. The script will try again tomorrow.");
      return false;
    }

    const status = response.getResponseCode();
    if (status === 202) return true;
    if (status === 200) {
      try {
        return JSON.parse(response.getContentText()).status === "duplicate";
      } catch (_error) {
        return false;
      }
    }
    if (status >= 500 && attempt === 0) {
      Utilities.sleep(RETRY_DELAY_MS);
      continue;
    }
    _log(
      status === 401
        ? "Wealth didn't accept the upload key. Create a new key in Settings and paste the new script."
        : "Wealth couldn't take a statement (status " + status + "). The script will try again tomorrow.",
    );
    return false;
  }
  return false;
}

function run() {
  const properties = PropertiesService.getScriptProperties();
  // A key pasted into the code wins over one saved by an earlier setup.
  const apiUrl =
    API_URL.indexOf("example.invalid") === -1 ? API_URL : properties.getProperty("API_URL") || API_URL;
  const uploadKey =
    UPLOAD_KEY.indexOf("wsk_") === 0 ? UPLOAD_KEY : properties.getProperty("UPLOAD_KEY") || UPLOAD_KEY;
  if (!apiUrl || !uploadKey || uploadKey.indexOf("wsk_") !== 0) {
    throw new Error("Add your Wealth upload key before running the script.");
  }

  const label = GmailApp.getUserLabelByName(IMPORTED_LABEL) || GmailApp.createLabel(IMPORTED_LABEL);
  const firstRun = properties.getProperty("FIRST_RUN_DONE") !== "yes";
  const days = firstRun ? FIRST_RUN_DAYS : DAILY_DAYS;
  const found = GmailApp.search(_searchQuery(days), 0, 100);
  const threads = firstRun ? _newestPerSender(found) : found;
  _log(
    threads.length
      ? "Found " + threads.length + " statement email(s) from the last " + days + " days."
      : "No new statement emails from CAMS, KFintech, MF Central, NSDL or CDSL in the last " +
          days +
          " days.",
  );
  let sent = 0;
  let failed = 0;
  let labelled = 0;

  threads.forEach(function (thread) {
    let foundPdf = false;
    let allSent = true;
    thread.getMessages().forEach(function (message) {
      message.getAttachments({ includeInlineImages: false, includeAttachments: true }).forEach(
        function (attachment) {
          if (!_isPdf(attachment)) return;
          foundPdf = true;
          if (_postPdf(message, attachment, apiUrl, uploadKey)) sent += 1;
          else {
            failed += 1;
            allSent = false;
          }
        },
      );
    });
    if (foundPdf && allSent) {
      thread.addLabel(label);
      labelled += 1;
    }
  });

  if (firstRun && failed === 0) properties.setProperty("FIRST_RUN_DONE", "yes");
  if (sent) _log("Sent " + sent + " statement PDF(s). Open Inbox in Wealth to enter the password.");
  return { sent: sent, labelled: labelled };
}

if (typeof module !== "undefined") {
  module.exports = { setup: setup, run: run, searchQuery: _searchQuery };
}
