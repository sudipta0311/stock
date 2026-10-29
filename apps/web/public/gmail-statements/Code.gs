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

function _searchQuery() {
  const senderSearch = KNOWN_SENDERS.map(function (sender) {
    return "from:" + sender;
  }).join(" ");
  // Gmail search writes a nested label "Wealth/Imported" as "wealth-imported".
  const labelSearch = IMPORTED_LABEL.toLowerCase().replace(/[\/\s]+/g, "-");
  return "has:attachment filename:pdf newer_than:45d -label:" + labelSearch + " {" + senderSearch + "}";
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
    return false;
  }
  return false;
}

function run() {
  const properties = PropertiesService.getScriptProperties();
  const apiUrl = properties.getProperty("API_URL") || API_URL;
  const uploadKey = properties.getProperty("UPLOAD_KEY") || UPLOAD_KEY;
  if (!apiUrl || !uploadKey || uploadKey.indexOf("wsk_") !== 0) {
    throw new Error("Add your Wealth upload key before running the script.");
  }

  const label = GmailApp.getUserLabelByName(IMPORTED_LABEL) || GmailApp.createLabel(IMPORTED_LABEL);
  const threads = GmailApp.search(_searchQuery(), 0, 100);
  let sent = 0;
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
          else allSent = false;
        },
      );
    });
    if (foundPdf && allSent) {
      thread.addLabel(label);
      labelled += 1;
    }
  });

  return { sent: sent, labelled: labelled };
}

if (typeof module !== "undefined") {
  module.exports = { setup: setup, run: run, searchQuery: _searchQuery };
}
