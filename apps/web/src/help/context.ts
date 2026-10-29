export function helpArticleForAssetType(assetType: string): string {
  if (assetType === "loan") return "loans";
  if (assetType === "epf" || assetType === "ppf" || assetType === "nps") {
    return "retirement-savings";
  }
  if (assetType === "po_scheme") return "post-office";
  if (assetType === "insurance") return "insurance";
  if (assetType === "gold") return "gold";
  if (assetType === "property") return "property";
  if (assetType === "us_investments") return "money-abroad";
  if (["stocks", "funds", "stock", "mutual_fund"].includes(assetType)) {
    return "shares-funds";
  }
  return "deposits";
}

export function helpArticleForPath(pathname: string): string {
  if (pathname.startsWith("/import")) return "uploading-statements";
  if (pathname.startsWith("/money/loan")) return "loans";
  if (pathname.startsWith("/money/holding")) return "shares-funds";
  if (pathname.startsWith("/money/add/")) {
    return helpArticleForAssetType(pathname.split("/")[3] ?? "");
  }
  if (pathname.startsWith("/money")) return "adding-money";
  if (pathname.startsWith("/plan")) return "your-plan";
  if (pathname.startsWith("/ideas")) return "investing-denmark";
  if (pathname.startsWith("/inbox")) return "notes-emails";
  if (pathname.startsWith("/settings")) return "privacy-safety";
  if (pathname.startsWith("/help")) return "getting-started";
  return "this-week";
}
