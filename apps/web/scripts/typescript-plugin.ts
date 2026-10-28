import ts from "typescript";
import type { Plugin } from "vite";

export function typescriptPlugin(): Plugin {
  return {
    name: "typescript-transpile",
    enforce: "pre",
    transform(source, id) {
      const fileName = id.split("?", 1)[0] ?? id;
      if (!/\.tsx?$/.test(fileName) || fileName.endsWith(".d.ts")) return null;
      const result = ts.transpileModule(source, {
        fileName,
        compilerOptions: {
          target: ts.ScriptTarget.ES2022,
          module: ts.ModuleKind.ESNext,
          jsx: ts.JsxEmit.ReactJSX,
          sourceMap: true,
          inlineSources: true,
        },
      });
      return {
        code: result.outputText,
        map: result.sourceMapText ? JSON.parse(result.sourceMapText) : null,
      };
    },
  };
}
