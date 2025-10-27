# Bun is a fast JavaScript all-in-one toolkit\|

[Bun v1.2 is here! Postgres, S3, better Node compatibility →](https://bun.com/blog/bun-v1.2)

Search`` `K`

Ask AI

![ai chat avatar](https://bun.com/logo_avatar.svg)

[Bun v1.2.19 is here! →](https://bun.com/blog/bun-v1.2.19)

Develop, test, run, and bundle JavaScript & TypeScript projects—all with Bun. Bun is an all-in-one JavaScript runtime & toolkit designed for speed, complete with a bundler, [test runner](https://bun.com/docs/cli/test), and Node.js-compatible [package manager](https://bun.com/package-manager). Bun aims for 100% Node.js compatibility.

Install Bun v1.2.19

Linux & macOSWindows

[View install script](https://bun.sh/install.ps1)

```bash
curl -fsSL https://bun.sh/install | bash
```

```bash
powershell -c "irm bun.sh/install.ps1 | iex"
```

USED BY

![Company logos using Bun](https://bun.com/images/logo-bar.png)

ExpressPostgresWebSocket

## Express.js 'hello world'

HTTP requests per second (Linux x64)

- bun: 59,026 requests per second

59,026

- deno: 25,335 requests per second

25,335

- node: 19,039 requests per second

19,039

[Bun\\
\\
v1.2](https://github.com/oven-sh/bun/blob/246936a7a4a0c5c8025a14072f49fde11d599a71/bench/express/express.mjs) [Deno\\
\\
v2.1.6](https://github.com/oven-sh/bun/blob/246936a7a4a0c5c8025a14072f49fde11d599a71/bench/express/express.mjs) [Node.js\\
\\
v23.6.0](https://github.com/oven-sh/bun/blob/246936a7a4a0c5c8025a14072f49fde11d599a71/bench/express/express.mjs)

[View benchmark](https://github.com/oven-sh/bun/tree/main/bench/express "$ bombardier <url> -d 5s")

## WebSocket chat server

Messages sent per second (Linux x64, 32 clients)

- bun: 2,536,227 messages sent per second

2,536,227

- deno: 1,320,525 messages sent per second

1,320,525

- node: 435,099 messages sent per second

435,099

[Bun.serve()\\
\\
v1.2](https://github.com/oven-sh/bun/blob/58417217d6c68c2fcf5169f0e5031b0ce9005bf3/bench/websocket-server/chat-server.bun.js) [Deno.serve()\\
\\
v1.2.6](https://github.com/oven-sh/bun/blob/58417217d6c68c2fcf5169f0e5031b0ce9005bf3/bench/websocket-server/chat-server.deno.mjs) [ws (Node.js)\\
\\
v23.6.0](https://github.com/oven-sh/bun/blob/58417217d6c68c2fcf5169f0e5031b0ce9005bf3/bench/websocket-server/chat-server.node.mjs)

[View benchmark](https://github.com/oven-sh/bun/tree/main/bench/websocket-server)

## Load a huge table

Queries per second. 100 rows x 100 parallel queries

- bun: 50,251 queries per second

50,251

- node: 14,398 queries per second

14,398

- deno: 11,821 queries per second

11,821

[Bun\\
\\
v1.2](https://github.com/oven-sh/bun/blob/main/bench/postgres/index.mjs) [Node.js\\
\\
v23.6.0](https://github.com/oven-sh/bun/blob/main/bench/postgres/index.mjs) [Deno\\
\\
v2.1.6](https://github.com/oven-sh/bun/blob/main/bench/postgres/index.mjs)

[View benchmark](https://github.com/oven-sh/bun/tree/main/bench/postgres)

## What's different about Bun?

Bun provides extensive builtin APIs and tooling

|                                                                                                                                                |                                                        |                                                                                      |                                        |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------ | -------------------------------------- |
| ### BuiltinCore Features<br>Essential runtime capabilities                                                                                     | ![Bun](https://bun.com/logo.svg)<br>Bun                | Node                                                                                 | Deno                                   |
| [Node.js compatibility\<br>\<br>Aiming to be a drop-in replacement for Node.js apps](https://bun.com/docs/runtime/nodejs-apis)                 |                                                        |                                                                                      |                                        |
| [Web Standard APIs\<br>\<br>Support for web standard APIs like `fetch`, `URL`, `EventTarget`, `Headers`, etc.](https://bun.com/docs/api/fetch) | Powered by WebCore (from WebKit/Safari)                |                                                                                      |                                        |
| [Native Addons\<br>\<br>Call C-compatible native code from JavaScript](https://bun.com/docs/api/ffi)                                           | Bun.ffi, NAPI, partial V8 C++ API                      |                                                                                      |                                        |
| [TypeScript\<br>\<br>First-class support, including `"paths"` `enum` `namespace`](https://bun.com/docs/runtime/typescript)                     |                                                        |                                                                                      |                                        |
| [JSX\<br>\<br>First-class support without configuration](https://bun.com/docs/runtime/jsx)                                                     |                                                        |                                                                                      |                                        |
| [Module loader plugins\<br>\<br>Plugin API for importing/requiring custom file types](https://bun.com/docs/runtime/plugins)                    | \`Bun.plugin\` works in browsers & Bun                 | 3 different loader APIs. Server-side only                                            |                                        |
| ### BuiltinAPIs<br>Built-in performance and native APIs designed for production                                                                | ![Bun](https://bun.com/logo.svg)<br>Bun                | Node                                                                                 | Deno                                   |
| [PostgresSQL driver\<br>\<br>High-performance cloud database access](https://bun.com/docs/api/sql)                                             | Fastest available                                      |                                                                                      |                                        |
| [SQLite driver\<br>\<br>High-performance local database access](https://bun.com/docs/api/sqlite)                                               | Fastest available                                      |                                                                                      |                                        |
| [S3 Cloud Storage driver\<br>\<br>High-performance S3-compatible object storage](https://bun.com/docs/api/s3)                                  | Fastest available                                      |                                                                                      |                                        |
| [Redis client\<br>\<br>Standard key/value store with optional persistence](https://bun.com/docs/api/redis)                                     |                                                        |                                                                                      |                                        |
| [WebSocket server (including pub/sub)\<br>\<br>Builtin WebSocket implementation](https://bun.com/docs/api/websockets)                          | \`Bun.serve()\`                                        |                                                                                      |                                        |
| [HTTP server\<br>\<br>Native HTTP & HTTPS server implementation](https://bun.com/docs/api/http)                                                | Bun.serve()                                            |                                                                                      |                                        |
| [HTTP router\<br>\<br>Route HTTP requests to different handlers with /:dynamic/ and /\*wildcard routes](https://bun.com/docs/api/http)         | Bun.serve({routes: {'/api/:path': (req) => { ... }}}}) |                                                                                      |                                        |
| [Single-file executables\<br>\<br>Bundle your app into a single file for easy deployment](https://bun.com/docs/bundler/executables)            | bun build --compile                                    | No native addons, embedded files, cross-compilation or bytecode. Multi-step process. | No native addons, no cross-compilation |
| ### BuiltinTooling<br>Built-in developer tooling                                                                                               | ![Bun](https://bun.com/logo.svg)<br>Bun                | Node                                                                                 | Deno                                   |
| [npm package management\<br>\<br>Install, manage, and publish npm-compatible dependencies](https://bun.com/docs/cli/install)                   |                                                        |                                                                                      | Limited features                       |
| [Bundler\<br>\<br>Build production-ready code for frontend & backend](https://bun.com/docs/bundler)                                            | Bun.build                                              |                                                                                      |                                        |
| [Cross-platform $ shell API\<br>\<br>Native bash-like shell for cross-platform shell scripting](https://bun.com/docs/runtime/shell)            | \`Bun.$\`                                              |                                                                                      | Requires 'dax'                         |
| [Jest-compatible test runner\<br>\<br>Testing library compatible with the most popular testing framework](https://bun.com/docs/cli/test)       | bun test                                               |                                                                                      |                                        |
| [Hot reloading (server)\<br>\<br>Reload your backend without disconnecting connections, preserving state](https://bun.com/docs/runtime/hot)    | bun --hot                                              |                                                                                      |                                        |
| [Monorepo support\<br>\<br>Install workspaces packages and run commands across workspaces](https://bun.com/docs/cli/filter)                    | bun run --filter=package-glob ...                      |                                                                                      |                                        |
| [Frontend Development Server\<br>\<br>Run modern frontend apps with a fully-featured dev server](https://bun.com/docs/bundler/html)            | bun ./index.html                                       |                                                                                      |                                        |
| Formatter & Linter<br>Built-in formatter and linter                                                                                            |                                                        |                                                                                      |                                        |
| ### BuiltinUtilities<br>APIs that make your life easier as a developer                                                                         | ![Bun](https://bun.com/logo.svg)<br>Bun                | Node                                                                                 | Deno                                   |
| [Password & Hashing APIs\<br>\<br>`bcrypt`, `argon2`, and non-cryptographic hash functions](https://bun.com/docs/api/hashing)                  | \`Bun.password\` & \`Bun.hash\`                        |                                                                                      |                                        |
| [String Width API\<br>\<br>Calculate the width of a string as displayed in the terminal](https://bun.com/docs/api/utils)                       | Bun.stringWidth                                        |                                                                                      |                                        |
| [Glob API\<br>\<br>Glob patterns for file matching](https://bun.com/docs/api/glob)                                                             | Bun.Glob                                               | fs.promises.glob                                                                     |                                        |
| [Semver API\<br>\<br>Compare and sort semver strings](https://bun.com/docs/api/semver)                                                         | Bun.semver                                             |                                                                                      |                                        |
| [CSS color conversion API\<br>\<br>Convert between CSS color formats](https://bun.com/docs/api/color)                                          | Bun.color                                              |                                                                                      |                                        |

$ bun run

## Bun is a JavaScript runtime.

Bun is a new JavaScript runtime built from scratch to serve the modern JavaScript ecosystem. It has three major design goals:

- **Speed.** Bun starts fast and runs fast. It extends JavaScriptCore, the performance-minded JS engine built for Safari. Fast start times mean fast apps and fast APIs.
- **Elegant APIs.** Bun provides a minimal set of highly-optimized APIs for performing common tasks, like starting an HTTP server and writing files.
- **Cohesive DX.** Bun is a complete toolkit for building JavaScript apps, including a package manager, test runner, and bundler.

Bun is designed as a drop-in replacement for Node.js. It natively implements hundreds of Node.js and Web APIs, including `fs`, `path`, `Buffer` and more.

The goal of Bun is to run most of the world's server-side JavaScript and provide tools to improve performance, reduce complexity, and multiply developer productivity.

[Drop-in Node.js compatibility\\
\\
Bun aims to be a drop-in replacement for Node.js. It implements Node's module resolution algorithm, globals like `Buffer` and `process`, and built-in modules like `fs` and `path`. Click to track Bun's progress towards full compatibility.](https://bun.com/docs/runtime/nodejs-apis)

Fast running performance

Bun extends the JavaScriptCore engine—the performance-minded JS engine built for Safari—with native-speed functionality implemented in Zig.

[Works with `node_modules`\\
\\
With Bun, you still use `package.json` to manage your dependencies. Use Bun's native npm client to see just how fast installing dependencies can be.](https://bun.com/docs/cli/install) [No more module madness\\
\\
Forget the complicated rules around CommonJS, ESM, file extensions, resolution priority, and `package.json` configurations. With Bun, it just works.](https://bun.com/docs/runtime/modules) [TypeScript\\
\\
TypeScript is a first-class citizen in Bun. Directly execute `.ts` and `.tsx` files. Bun respects your settings configured in `tsconfig.json`, including `"paths"`, `"jsx"`, and more.](https://bun.com/docs/runtime/typescript) [Web-standard APIs\\
\\
Bun implements the Web-standard APIs you know and love, including `fetch`, `ReadableStream`, `Request`, `Response`, `WebSocket`, and `FormData`.](https://bun.com/docs/runtime/web-apis) [JSX\\
\\
JSX just works. Bun internally transpiles JSX syntax to vanilla JavaScript. Like TypeScript itself, Bun assumes React by default but respects custom JSX transforms defined in `tsconfig.json`.](https://bun.com/docs/runtime/jsx) [Watch mode\\
\\
The `bun run` CLI provides a smart `--watch` flag that automatically restarts the process when any imported file changes.](https://bun.com/docs/runtime/hot) [Cross-platform shell scripts\\
\\
The `Bun.$` API implements a cross-platform bash-like interpreter, shell, and coreutils. This makes it easy to run shell scripts from JavaScript for devops tasks.](https://bun.com/docs/runtime/shell)

## The APIs you need. Baked in.

Start an HTTP server

Start a WebSocket server

Read and write files

Hash a password

Bundle for the browser

Write a test

File system routing

Query a SQLite database

Run a shell script

Call a C function

index.tsx

```typescript
import { sql, serve } from "bun";

const server = serve({
  port: 3000,
  routes: {
    "/": () => new Response("Welcome to Bun!"),
    "/api/users": async (req) => {
      const users = await sql`SELECT * FROM users LIMIT 10`;
      return Response.json({ users });
    },
  },
});

console.log(`Listening on localhost:${server.port}`);
```

index.tsx

```typescript
const server = Bun.serve<{ authToken: string }>({
  fetch(req, server) {
    // use a library to parse cookies
    const cookies = parseCookies(req.headers.get("Cookie"));
    server.upgrade(req, {
      data: { authToken: cookies["X-Token"] },
    });
  },
  websocket: {
    // handler called when a message is received
    async message(ws, message) {
      console.log(`Received: ${message}`);
      const user = getUserFromToken(ws.data.authToken);
      await db.Message.insert({
        message: String(message),
        userId: user.id,
      });
    },
  },
});

console.log(`Listening on localhost:${server.port}`);
```

index.tsx

```typescript
const file = Bun.file(import.meta.dir + "/package.json"); // BunFile

const pkg = await file.json(); // BunFile extends Blob
pkg.name = "my-package";
pkg.version = "1.0.0";

await Bun.write(file, JSON.stringify(pkg, null, 2));
```

index.tsx

```typescript
const password = "super-secure-pa$$word";

const hash = await Bun.password.hash(password);
// => $argon2id$v=19$m=65536,t=2,p=1$tFq+9AVr1bfPxQdh...

const isMatch = await Bun.password.verify(password, hash);
// => true
```

bundle.tsx

```typescript
await Bun.build({
  entrypoints: ["./index.tsx"],
  outdir: "./build",
  minify: true,
  plugins: [
    /* ... */
  ],
});
```

index.tsx

```typescript
import { describe, expect, test, beforeAll } from "bun:test";

beforeAll(() => {
  // setup tests
});

describe("math", () => {
  test("addition", () => {
    expect(2 + 2).toBe(4);
    expect(7 + 13).toMatchSnapshot();
  });
});
```

index.tsx

```typescript
const router = new Bun.FileSystemRouter({
  style: "nextjs",
  dir: "/path/to/pages",
});

const match = router.match("/blog/my-cool-post");
match.filePath; // "/path/to/pages/blog/[slug].tsx",
match.kind; // "dynamic"
match.params; // { slug: "my-cool-post" }
```

index.tsx

```typescript
import { Database } from "bun:sqlite";

const db = new Database("db.sqlite");

console.log(db.query("SELECT 1 as x").get());
// { x: 1 }
```

index.tsx

```typescript
import { $ } from "bun";

// Run a shell command (also works on Windows!)
await $`echo "Hello, world!"`;

const response = await fetch("https://example.com");

// Pipe the response body to gzip
const data = await $`gzip < ${response}`.arrayBuffer();
```

index.tsx

```typescript
import { dlopen, FFIType, suffix } from "bun:ffi";

// `suffix` is either "dylib", "so", or "dll" depending on the platform
const path = `libsqlite3.${suffix}`;

const {
  symbols: {
    sqlite3_libversion, // the function to call
  },
} = dlopen(path, {
  sqlite3_libversion: {
    args: [], // no arguments
    returns: FFIType.cstring, // returns a string
  },
});

console.log(`SQLite 3 version: ${sqlite3_libversion()}`);
```

formerly Twitter

Production

> "Bun is at the heart of one of our newest infrastructure projects \[...\]
>
> in production, powering our sports and upcoming news products" [→](https://x.com/shlomiatar/status/1874234143971958831)

![Avatar](https://bun.com/images/shlomi.png)

Shlomi Atar

Engineering Team at X

$ bun install

## Bun is an npm-compatible package manager.

Bun

00.36s

pnpm

_17x slower_

01.53s

npm

_29x slower_

01.53s

Yarn

_33x slower_

01.53s

Installing dependencies from cache for a Remix app.

[View benchmark](https://github.com/oven-sh/bun/tree/main/bench/install)

[Node.js compatible\\
\\
Bun still installs your dependencies into `node_modules` like `npm` and other package managers—it just does it faster. You don't need to use the Bun runtime to use Bun as a package manager.](https://bun.com/docs/cli/install) [Crazy fast\\
\\
Bun uses the fastest system calls available on each operating system to make installs faster than you'd think possible.](https://github.com/oven-sh/bun/tree/main/bench/install) [Workspaces\\
\\
Workspaces are supported out of the box. Bun reads the `workspaces` key from your `package.json` and installs dependencies for your whole monorepo.](https://bun.com/docs/install/workspaces) [Global install cache\\
\\
Download once, install anywhere. Bun only downloads a particular version of a package from npm once; future installations will copy it from the cache.](https://bun.com/docs/install/cache) [Security by default\\
\\
Unlike other package managers, Bun doesn't execute `postinstall` scripts by default. Popular packages are automatically allow-listed; others can be added to the `trustedDependencies` in your `package.json`.](https://bun.com/docs/cli/install#lifecycle-scripts) [Cross-platform package.json scripts\\
\\
On Windows, package.json scripts are powered by the Bun Shell. It's now safe to delete `cross-env`, `rimraf` and `node-which`.](https://bun.com/docs/runtime/shell) [Familiar API\\
\\
Bun's CLI uses commands and flags that will feel familiar to any users of `npm`, `pnpm`, or `yarn`.](https://bun.com/docs/cli/install) [Reads .npmrc & package-lock.json\\
\\
Migrate from npm without changing dependency versions. Try bun install secretly without telling your coworkers.](https://bun.com/docs/cli/install)

Replace `yarn` with `bun install` to get 30x faster package installs.

[Try it](https://bun.com/docs/cli/install)

$ bun test

## Bun is a test runner that makes the rest look like test walkers.

Bun

00.23s

Vitest

_5x slower_

01.53s

Jest+SWC

_8x slower_

01.53s

Jest+tsjest

_18x slower_

01.53s

Jest+Babel

_20x slower_

01.53s

Running the test suite for [Zod](https://github.com/colinhacks/zod)

[View benchmark](https://gist.github.com/colinhacks/e3095172a883a9ee9c972caf33b5f54f)

[Jest-compatible syntax\\
\\
Bun provides a Jest-style `expect()` API. Switch to `bun test` with no code changes.](https://bun.com/docs/test/writing) [Crazy fast\\
\\
Bun's fast startup times shine in the test runner. You won't believe how much faster your tests will run.](https://twitter.com/jarredsumner/status/1542824445810642946) [Lifecycle hooks\\
\\
Run setup and teardown code per-test with `beforeEach`/ `afterEach` or per-file with `beforeAll`/ `afterAll`.](https://bun.com/docs/test/lifecycle) [ESM, TypeScript & JSX just work\\
\\
Zero configuration needed to test TypeScript, ESM, and JSX files.](https://bun.com/docs/cli/test) [Snapshot testing\\
\\
Full support for on-disk snapshot testing with `.toMatchSnapshot()`. Overwrite snapshots with the `--update-snapshots` flag.](https://bun.com/docs/test/snapshots) [DOM APIs\\
\\
Simulate DOM and browser APIs in your tests using happy-dom.](https://bun.com/docs/test/dom) [Watch mode\\
\\
Use the `--watch` flag to re-run tests when files change using Bun's instantaneous watch mode.](https://bun.com/docs/test/hot) [Function mocks\\
\\
Mock functions with `mock()` or spy on methods with `spyOn()`.](https://bun.com/docs/test/mocks)

Replace `jest` with `bun test` to run your tests 10-30x faster.

[Try it](https://bun.com/docs/cli/test)

## Bun is a complete JavaScript toolkitEverything you need to ship & maintain your app, built-in

|                                                                                                                                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ### Builtin Core Features<br>Essential runtime capabilities                                                                                                                                                             |
| [Node.js compatibility\<br>\<br>Bun aims to be a drop-in replacement for Node.js apps\<br>\<br>View Docs](https://bun.com/docs/runtime/nodejs-apis)                                                                     |
| [Web Standard APIspowered by WebCore (from WebKit/Safari)\<br>\<br>Support for web standard APIs like `fetch`, `URL`, `EventTarget`, `Headers`, etc.\<br>\<br>View Docs](https://bun.com/docs/api/fetch)                |
| [TypeScript\<br>\<br>First-class support, including `"paths"` `enum` `namespace`\<br>\<br>View Docs](https://bun.com/docs/runtime/typescript)                                                                           |
| [JSX\<br>\<br>First-class support without configuration\<br>\<br>View Docs](https://bun.com/docs/runtime/jsx)                                                                                                           |
| [Module loader plugins`Bun.plugin` works in browsers & Bun\<br>\<br>Plugin API for importing/requiring custom file types\<br>\<br>View Docs](https://bun.com/docs/runtime/plugins)                                      |
| ### Builtin APIs<br>Built-in performance and native APIs designed for production                                                                                                                                        |
| [PostgresSQL driver`Bun.sql` Fastest available to JavaScript\<br>\<br>High-performance cloud database access\<br>\<br>View Docs](https://bun.com/docs/api/sql)                                                          |
| [SQLite driver`bun:sqlite` Fastest available to JavaScript\<br>\<br>High-performance local database access\<br>\<br>View Docs](https://bun.com/docs/api/sqlite)                                                         |
| [S3 Cloud Storage driver`Bun.s3` Fastest available to JavaScript\<br>\<br>High-performance S3-compatible object storage\<br>\<br>View Docs](https://bun.com/docs/api/s3)                                                |
| [Redis client`Bun.redis`\<br>\<br>Standard key/value store with optional persistence\<br>\<br>View Docs](https://bun.com/docs/api/redis)                                                                                |
| [WebSocket server`Bun.serve()`\<br>\<br>Builtin WebSocket implementation\<br>\<br>View Docs](https://bun.com/docs/api/websockets)                                                                                       |
| [HTTP server`Bun.serve()`\<br>\<br>Native HTTP & HTTPS server implementation\<br>\<br>View Docs](https://bun.com/docs/api/http)                                                                                         |
| [Single-file executables`bun build --compile`\<br>\<br>Bundle your app into a single file for easy deployment\<br>\<br>View Docs](https://bun.com/docs/bundler/executables)                                             |
| [HTTP Router`Bun.serve({routes: {'/api/:path': async (req) => { ... }}}})`\<br>\<br>Route HTTP requests to different handlers with /:dynamic/ and /\*wildcard routes\<br>\<br>View Docs](https://bun.com/docs/api/http) |
| ### Builtin Tooling<br>Built-in developer tooling                                                                                                                                                                       |
| [Package Manager`bun install`\<br>\<br>Install, manage, and publish npm-compatible dependencies\<br>\<br>View Docs](https://bun.com/docs/cli/install)                                                                   |
| [Bundler`Bun.build`\<br>\<br>Build production-ready code for frontend & backend\<br>\<br>View Docs](https://bun.com/docs/bundler)                                                                                       |
| [Cross-platform shell API`Bun.$`\<br>\<br>Native bash-like shell for cross-platform shell scripting\<br>\<br>View Docs](https://bun.com/docs/runtime/shell)                                                             |
| [Jest-compatible test runner`bun test`\<br>\<br>Testing library compatible with the most popular testing framework\<br>\<br>View Docs](https://bun.com/docs/cli/test)                                                   |
| [Hot reloading`bun --hot`\<br>\<br>Reload your backend without disconnecting connections, preserving state\<br>\<br>View Docs](https://bun.com/docs/runtime/hot)                                                        |
| [Monorepo support`bun run --filter=package-glob ...`\<br>\<br>Install workspaces packages and run commands across workspaces\<br>\<br>View Docs](https://bun.com/docs/cli/filter)                                       |
| [Frontend Development Server`bun ./index.html`\<br>\<br>Run modern frontend apps with a fully-featured dev server\<br>\<br>View Docs](https://bun.com/docs/bundler/html)                                                |
| ### Builtin Utilities<br>APIs that make your life easier as a developer                                                                                                                                                 |
| [Password & Hashing`Bun.password` & `Bun.hash`\<br>\<br>bcrypt, argon2, and non-cryptographic hash functions\<br>\<br>View Docs](https://bun.com/docs/api/hashing)                                                      |
| [String Width API`Bun.stringWidth`\<br>\<br>Calculate the width of a string as displayed in the terminal\<br>\<br>View Docs](https://bun.com/docs/api/utils)                                                            |
| [CSS color conversion`Bun.color`\<br>\<br>Convert between CSS color formats\<br>\<br>View Docs](https://bun.com/docs/api/color)                                                                                         |
| [Glob API`Bun.Glob`\<br>\<br>Glob patterns for file matching\<br>\<br>View Docs](https://bun.com/docs/api/glob)                                                                                                         |
| [Semver API`Bun.semver`\<br>\<br>Compare and sort semver strings\<br>\<br>View Docs](https://bun.com/docs/api/semver)                                                                                                   |
| [Foreign Function Interface`bun:ffi`, NAPI, partial V8 C++ API\<br>\<br>Call C-compatible native code from JavaScript\<br>\<br>View Docs](https://bun.com/docs/api/ffi)                                                 |

1

Install Bun

```
curl -fsSL https://bun.sh/install | bash
```

```
powershell -c "irm bun.sh/install.ps1 | iex"
```

2

Write your code

index.tsx

```
const server = Bun.serve({
  port: 3000,
  fetch(request) {
    return new Response("Welcome to Bun!");
  },
});

console.log(`Listening on localhost:${server.port}`);
```

3

Run the file

```
bun index.tsx
```

```
bun index.tsx
```

[Install Bun](https://bun.com/docs/installation)

[Quick start](https://bun.com/docs/quickstart)

## Developers love Bun.

Sainder

Jan 17

@Sainder_Pradipt

Bun

Lic

Jan 18

@Lik228

bun

Martin Navrátil

Jan 17

@martin_nav\_

Bun....

SaltyAom

Jan 17

@saltyAom

bun

reaxios

Jan 17

@reaxios

bun install bun

kyge

Jan 17

@0xkyge

bun

James Landrum

Jan 17

@JamesRLandrum

Node

orlowdev

Jan 17

@orlowdev

Yeah, bun, but my code does not have dependencies.

hola

Jan 17

@jdggggyujhbc

bun

std::venom

Jan 17

@std_venom

Bun

tiago

Jan 19

@tiagorangel23

should have used Bun instead of npm

46officials

Jan 19

@46officials

Bun

yuki

Jan 19

@staticdots

Bun

Stefan

Jan 17

@stefangarofalo

Bun

Samuel

Jan 17

@samueldans0

Bun always

Divin Prince

Jan 17

@divinprnc

Yeah Bun

Gibson

Jan 16

@GibsonSMurray

bun

Oggie Sutrisna

Jan 16

@oggiesutrisna

bun

emanon

Jan 16

@0x_emanon

✅ bun

yuki

Jan 16

@staticdots

bun

SpiritBear

Jan 16

@0xSpiritBear

bun

Ayu

Jan 12

@Ayuu2809

Bun good 🧅

Hirbod

Jan 19

@hirbod_dev

For everything. Yes. I even run with bunx expo run:ios etc

Luis Paolini

Jan 18

@DigitalLuiggi

Jus use @bunjavascript

buraks

Jan 18

@buraks\_\_\_\_

I use bun patch and I love it!

fahadali

Jan 8

@fahadali503

Bun

Aiden Bai

Jan 1

@aidenybai

2025 will be the year of JS/TS and @bunjavascript is why

Catalin

Jan 1

@catalinmpit

Bun is goated

MadMax

Jan 3

@dr\_\_madmax

@bunjavascript is yet to get enough appreciation it deserves.

Baggi/e

Jan 3

@ManiSohi

Performant TS/JS backend needs more love
Elysia for the win

Michael Feldstein

Dec 18

@msfeldstein

holy shit bun is the solution to spending all day mucking around with typescript/module/commonjs/import bullshit and just running scripts

Sainder

Jan 17

@Sainder_Pradipt

Bun

Lic

Jan 18

@Lik228

bun

Martin Navrátil

Jan 17

@martin_nav\_

Bun....

SaltyAom

Jan 17

@saltyAom

bun

reaxios

Jan 17

@reaxios

bun install bun

kyge

Jan 17

@0xkyge

bun

James Landrum

Jan 17

@JamesRLandrum

Node

orlowdev

Jan 17

@orlowdev

Yeah, bun, but my code does not have dependencies.

hola

Jan 17

@jdggggyujhbc

bun

std::venom

Jan 17

@std_venom

Bun

tiago

Jan 19

@tiagorangel23

should have used Bun instead of npm

46officials

Jan 19

@46officials

Bun

yuki

Jan 19

@staticdots

Bun

Stefan

Jan 17

@stefangarofalo

Bun

Samuel

Jan 17

@samueldans0

Bun always

Divin Prince

Jan 17

@divinprnc

Yeah Bun

Gibson

Jan 16

@GibsonSMurray

bun

Oggie Sutrisna

Jan 16

@oggiesutrisna

bun

emanon

Jan 16

@0x_emanon

✅ bun

yuki

Jan 16

@staticdots

bun

SpiritBear

Jan 16

@0xSpiritBear

bun

Ayu

Jan 12

@Ayuu2809

Bun good 🧅

Hirbod

Jan 19

@hirbod_dev

For everything. Yes. I even run with bunx expo run:ios etc

Luis Paolini

Jan 18

@DigitalLuiggi

Jus use @bunjavascript

buraks

Jan 18

@buraks\_\_\_\_

I use bun patch and I love it!

fahadali

Jan 8

@fahadali503

Bun

Aiden Bai

Jan 1

@aidenybai

2025 will be the year of JS/TS and @bunjavascript is why

Catalin

Jan 1

@catalinmpit

Bun is goated

MadMax

Jan 3

@dr\_\_madmax

@bunjavascript is yet to get enough appreciation it deserves.

Baggi/e

Jan 3

@ManiSohi

Performant TS/JS backend needs more love
Elysia for the win

Michael Feldstein

Dec 18

@msfeldstein

holy shit bun is the solution to spending all day mucking around with typescript/module/commonjs/import bullshit and just running scripts

## Learn by example.

Our guides break down how to perform common tasks with Bun.

[Ecosystem\\
\\
Build a frontend using Vite and Bun\\
\\
View guide](https://bun.com/guides/ecosystem/vite) [Runtime\\
\\
Install TypeScript declarations for Bun\\
\\
View guide](https://bun.com/guides/runtime/typescript) [Streams\\
\\
Convert a ReadableStream to a string with Bun\\
\\
View guide](https://bun.com/guides/streams/to-string)

Ecosystem

[Use EdgeDB with Bun](https://bun.com/guides/ecosystem/edgedb)

[Use Prisma with Bun](https://bun.com/guides/ecosystem/prisma)

[Create a Discord bot](https://bun.com/guides/ecosystem/discordjs)

[Add Sentry to a Bun app](https://bun.com/guides/ecosystem/sentry)

[Use Drizzle ORM with Bun](https://bun.com/guides/ecosystem/drizzle)

[Build a React app with Bun](https://bun.com/guides/ecosystem/react)

[Run Bun as a daemon with PM2](https://bun.com/guides/ecosystem/pm2)

[Build an app with Nuxt and Bun](https://bun.com/guides/ecosystem/nuxt)

[Build an app with Qwik and Bun](https://bun.com/guides/ecosystem/qwik)

[Build an app with Astro and Bun](https://bun.com/guides/ecosystem/astro)

[Build an app with Remix and Bun](https://bun.com/guides/ecosystem/remix)

[Run Bun as a daemon with systemd](https://bun.com/guides/ecosystem/systemd)

[Build an app with Next.js and Bun](https://bun.com/guides/ecosystem/nextjs)

[Deploy a Bun application on Render](https://bun.com/guides/ecosystem/render)

[Build an app with SvelteKit and Bun](https://bun.com/guides/ecosystem/sveltekit)

[Build a frontend using Vite and Bun](https://bun.com/guides/ecosystem/vite)

[Build an app with SolidStart and Bun](https://bun.com/guides/ecosystem/solidstart)

[Use Neon Postgres through Drizzle ORM](https://bun.com/guides/ecosystem/neon-drizzle)

[Build an HTTP server using Hono and Bun](https://bun.com/guides/ecosystem/hono)

[Use Neon's Serverless Postgres with Bun](https://bun.com/guides/ecosystem/neon-serverless-postgres)

[Build an HTTP server using Elysia and Bun](https://bun.com/guides/ecosystem/elysia)

[Containerize a Bun application with Docker](https://bun.com/guides/ecosystem/docker)

[Build an HTTP server using Express and Bun](https://bun.com/guides/ecosystem/express)

[Server-side render (SSR) a React component](https://bun.com/guides/ecosystem/ssr-react)

[Build an HTTP server using StricJS and Bun](https://bun.com/guides/ecosystem/stric)

[Read and write data to MongoDB using Mongoose and Bun](https://bun.com/guides/ecosystem/mongoose)

HTMLRewriter

[Extract links from a webpage using HTMLRewriter](https://bun.com/guides/html-rewriter/extract-links)

[Extract social share images and Open Graph tags](https://bun.com/guides/html-rewriter/extract-social-meta)

HTTP

[Common HTTP server usage](https://bun.com/guides/http/server)

[Hot reload an HTTP server](https://bun.com/guides/http/hot)

[Write a simple HTTP server](https://bun.com/guides/http/simple)

[Start a cluster of HTTP servers](https://bun.com/guides/http/cluster)

[Configure TLS on an HTTP server](https://bun.com/guides/http/tls)

[Send an HTTP request using fetch](https://bun.com/guides/http/fetch)

[Proxy HTTP requests using fetch()](https://bun.com/guides/http/proxy)

[Stream a file as an HTTP Response](https://bun.com/guides/http/stream-file)

[Upload files via HTTP using FormData](https://bun.com/guides/http/file-uploads)

[fetch with unix domain sockets in Bun](https://bun.com/guides/http/fetch-unix)

[Streaming HTTP Server with Async Iterators](https://bun.com/guides/http/stream-iterator)

[Streaming HTTP Server with Node.js Streams](https://bun.com/guides/http/stream-node-streams-in-bun)

Package manager

[Add a dependency](https://bun.com/guides/install/add)

[Add a Git dependency](https://bun.com/guides/install/add-git)

[Add a peer dependency](https://bun.com/guides/install/add-peer)

[Add a tarball dependency](https://bun.com/guides/install/add-tarball)

[Add a trusted dependency](https://bun.com/guides/install/trusted)

[Add an optional dependency](https://bun.com/guides/install/add-optional)

[Add a development dependency](https://bun.com/guides/install/add-dev)

[Using bun install with Artifactory](https://bun.com/guides/install/jfrog-artifactory)

[Generate a yarn-compatible lockfile](https://bun.com/guides/install/yarnlock)

[Migrate from npm install to bun install](https://bun.com/guides/install/from-npm-install-to-bun-install)

[Configuring a monorepo using workspaces](https://bun.com/guides/install/workspaces)

[Install a package under a different name](https://bun.com/guides/install/npm-alias)

[Configure git to diff Bun's lockb lockfile](https://bun.com/guides/install/git-diff-bun-lockfile)

[Install dependencies with Bun in GitHub Actions](https://bun.com/guides/install/cicd)

[Override the default npm registry for bun install](https://bun.com/guides/install/custom-registry)

[Using bun install with an Azure Artifacts npm registry](https://bun.com/guides/install/azure-artifacts)

[Configure a private registry for an organization scope with bun install](https://bun.com/guides/install/registry-scope)

Processes

[Read from stdin](https://bun.com/guides/process/stdin)

[Listen for CTRL+C](https://bun.com/guides/process/ctrl-c)

[Listen to OS signals](https://bun.com/guides/process/os-signals)

[Spawn a child process](https://bun.com/guides/process/spawn)

[Parse command-line arguments](https://bun.com/guides/process/argv)

[Read stderr from a child process](https://bun.com/guides/process/spawn-stderr)

[Read stdout from a child process](https://bun.com/guides/process/spawn-stdout)

[Get the process uptime in nanoseconds](https://bun.com/guides/process/nanoseconds)

[Spawn a child process and communicate using IPC](https://bun.com/guides/process/ipc)

Reading files

[Read a JSON file](https://bun.com/guides/read-file/json)

[Check if a file exists](https://bun.com/guides/read-file/exists)

[Read a file to a Buffer](https://bun.com/guides/read-file/buffer)

[Read a file as a string](https://bun.com/guides/read-file/string)

[Get the MIME type of a file](https://bun.com/guides/read-file/mime)

[Read a file to a Uint8Array](https://bun.com/guides/read-file/uint8array)

[Read a file to an ArrayBuffer](https://bun.com/guides/read-file/arraybuffer)

[Watch a directory for changes](https://bun.com/guides/read-file/watch)

[Read a file as a ReadableStream](https://bun.com/guides/read-file/stream)

Runtime

[Delete files](https://bun.com/guides/runtime/delete-file)

[Delete directories](https://bun.com/guides/runtime/delete-directory)

[Import a JSON file](https://bun.com/guides/runtime/import-json)

[Import a TOML file](https://bun.com/guides/runtime/import-toml)

[Run a Shell Command](https://bun.com/guides/runtime/shell)

[Re-map import paths](https://bun.com/guides/runtime/tsconfig-paths)

[Set a time zone in Bun](https://bun.com/guides/runtime/timezone)

[Set environment variables](https://bun.com/guides/runtime/set-env)

[Import a HTML file as text](https://bun.com/guides/runtime/import-html)

[Read environment variables](https://bun.com/guides/runtime/read-env)

[Build-time constants with --define](https://bun.com/guides/runtime/build-time-constants)

[Debugging Bun with the web debugger](https://bun.com/guides/runtime/web-debugger)

[Install and run Bun in GitHub Actions](https://bun.com/guides/runtime/cicd)

[Install TypeScript declarations for Bun](https://bun.com/guides/runtime/typescript)

[Debugging Bun with the VS Code extension](https://bun.com/guides/runtime/vscode-debugger)

[Inspect memory usage using V8 heap snapshots](https://bun.com/guides/runtime/heap-snapshot)

[Define and replace static globals & constants](https://bun.com/guides/runtime/define-constant)

[Codesign a single-file JavaScript executable on macOS](https://bun.com/guides/runtime/codesign-macos-executable)

Streams

[Convert a ReadableStream to JSON](https://bun.com/guides/streams/to-json)

[Convert a Node.js Readable to JSON](https://bun.com/guides/streams/node-readable-to-json)

[Convert a ReadableStream to a Blob](https://bun.com/guides/streams/to-blob)

[Convert a Node.js Readable to a Blob](https://bun.com/guides/streams/node-readable-to-blob)

[Convert a ReadableStream to a Buffer](https://bun.com/guides/streams/to-buffer)

[Convert a ReadableStream to a string](https://bun.com/guides/streams/to-string)

[Convert a Node.js Readable to a string](https://bun.com/guides/streams/node-readable-to-string)

[Convert a ReadableStream to a Uint8Array](https://bun.com/guides/streams/to-typedarray)

[Convert a ReadableStream to an ArrayBuffer](https://bun.com/guides/streams/to-arraybuffer)

[Convert a Node.js Readable to an Uint8Array](https://bun.com/guides/streams/node-readable-to-uint8array)

[Convert a Node.js Readable to an ArrayBuffer](https://bun.com/guides/streams/node-readable-to-arraybuffer)

[Convert a ReadableStream to an array of chunks](https://bun.com/guides/streams/to-array)

Test runner

[Mock functions in `bun test`](https://bun.com/guides/test/mock-functions)

[Spy on methods in `bun test`](https://bun.com/guides/test/spy-on)

[Using Testing Library with Bun](https://bun.com/guides/test/testing-library)

[Update snapshots in `bun test`](https://bun.com/guides/test/update-snapshots)

[Run tests in watch mode with Bun](https://bun.com/guides/test/watch-mode)

[Use snapshot testing in `bun test`](https://bun.com/guides/test/snapshot)

[Bail early with the Bun test runner](https://bun.com/guides/test/bail)

[Skip tests with the Bun test runner](https://bun.com/guides/test/skip-tests)

[Migrate from Jest to Bun's test runner](https://bun.com/guides/test/migrate-from-jest)

[Run your tests with the Bun test runner](https://bun.com/guides/test/run-tests)

[Set the system time in Bun's test runner](https://bun.com/guides/test/mock-clock)

[Write browser DOM tests with Bun and happy-dom](https://bun.com/guides/test/happy-dom)

[Set a per-test timeout with the Bun test runner](https://bun.com/guides/test/timeout)

[Mark a test as a "todo" with the Bun test runner](https://bun.com/guides/test/todo-tests)

[Re-run tests multiple times with the Bun test runner](https://bun.com/guides/test/rerun-each)

[Set a code coverage threshold with the Bun test runner](https://bun.com/guides/test/coverage-threshold)

[Generate code coverage reports with the Bun test runner](https://bun.com/guides/test/coverage)

[import, require, and test Svelte components with bun test](https://bun.com/guides/test/svelte-test)

Utilities

[Hash a password](https://bun.com/guides/util/hash-a-password)

[Generate a UUID](https://bun.com/guides/util/javascript-uuid)

[Escape an HTML string](https://bun.com/guides/util/escape-html)

[Get the current Bun version](https://bun.com/guides/util/version)

[Encode and decode base64 strings](https://bun.com/guides/util/base64)

[Check if two objects are deeply equal](https://bun.com/guides/util/deep-equals)

[Detect when code is executed with Bun](https://bun.com/guides/util/detect-bun)

[Get the directory of the current file](https://bun.com/guides/util/import-meta-dir)

[Get the file name of the current file](https://bun.com/guides/util/import-meta-file)

[Convert a file URL to an absolute path](https://bun.com/guides/util/file-url-to-path)

[Compress and decompress data with gzip](https://bun.com/guides/util/gzip)

[Convert an absolute path to a file URL](https://bun.com/guides/util/path-to-file-url)

[Get the path to an executable bin file](https://bun.com/guides/util/which-path-to-executable-bin)

[Sleep for a fixed number of milliseconds](https://bun.com/guides/util/sleep)

[Compress and decompress data with DEFLATE](https://bun.com/guides/util/deflate)

[Get the absolute path of the current file](https://bun.com/guides/util/import-meta-path)

[Check if the current file is the entrypoint](https://bun.com/guides/util/entrypoint)

[Get the absolute path to the current entrypoint](https://bun.com/guides/util/main)

WebSocket

[Build a simple WebSocket server](https://bun.com/guides/websocket/simple)

[Enable compression for WebSocket messages](https://bun.com/guides/websocket/compression)

[Build a publish-subscribe WebSocket server](https://bun.com/guides/websocket/pubsub)

[Set per-socket contextual data on a WebSocket](https://bun.com/guides/websocket/context)

Writing files

[Delete a file](https://bun.com/guides/write-file/unlink)

[Write to stdout](https://bun.com/guides/write-file/stdout)

[Write a Blob to a file](https://bun.com/guides/write-file/blob)

[Write a file to stdout](https://bun.com/guides/write-file/cat)

[Append content to a file](https://bun.com/guides/write-file/append)

[Write a string to a file](https://bun.com/guides/write-file/basic)

[Write a file incrementally](https://bun.com/guides/write-file/filesink)

[Write a Response to a file](https://bun.com/guides/write-file/response)

[Copy a file to another location](https://bun.com/guides/write-file/file-cp)

[Write a ReadableStream to a file](https://bun.com/guides/write-file/stream)

Powered by

[![inkeep search icon](https://uploads-ssl.webflow.com/63fd919a913cf54ca2d02cda/642ea6563549ced1bb379fea_inkeep-icon-medium-gray.svg)inkeep](https://www.inkeep.com/)

![ai chat avatar](https://bun.com/logo_avatar.svg)

Hi!

I'm an AI assistant trained on documentation, GitHub issues, and other content.

Ask me anything about `Bun`.

### Popular Questions

Can I use Bun with my existing Node.js project?

How is Bun faster than Node.js? How can I benchmark it?

Do I still need a bundler or TypeScript compiler?

---

Powered by

[![inkeep search icon](https://uploads-ssl.webflow.com/63fd919a913cf54ca2d02cda/642ea6563549ced1bb379fea_inkeep-icon-medium-gray.svg)inkeep](https://www.inkeep.com/)

Get help

[Discord](https://bun.com/discord)

[Migration help for organizations](https://t.co/0CA0Neqgts)
