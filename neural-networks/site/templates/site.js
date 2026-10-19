(function () {
  // Progress lives only in this browser. The storage keys and module numbers never change, so
  // learners keep what they completed when modules are added; MODULES comes from the build.
  var KEY = "nn1001-done", OPENED_KEY = "nn1001-opened";
  var MODULES = window.NN1001_MODULES || ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"];
  var ADDED_LATER = window.NN1001_ADDED_LATER || [];   // modules published after the course launched
  var T = window.NN1001_I18N || {};
  function fmt(key, vars, fallback) {
    var s = T[key] || fallback;
    Object.keys(vars).forEach(function (k) { s = s.split("{" + k + "}").join(vars[k]); });
    return s;
  }

  function loadList(key) {
    try {
      var list = JSON.parse(localStorage.getItem(key) || "[]");
      // ignore anything that is not a current module (old or hand-edited entries would skew the count)
      return Array.isArray(list) ? list.filter(function (m, i) { return MODULES.indexOf(m) >= 0 && list.indexOf(m) === i; }) : [];
    } catch (e) { return []; }
  }
  function saveList(key, list) {
    try { localStorage.setItem(key, JSON.stringify(list)); } catch (e) {}
  }
  function load() { return loadList(KEY); }
  function save(list) { saveList(KEY, list); }

  // a lesson page records that its module was opened, so the syllabus can show "in progress"
  var lessonBtn = document.querySelector(".complete-btn");
  // the modules that existed when this learner arrived; learners from before this list was kept
  // (progress but no list) arrived before every module in ADDED_LATER was published
  var KNOWN_KEY = "nn1001-known", known = loadList(KNOWN_KEY);
  if (!known.length) {
    var returning = load().length > 0 || loadList(OPENED_KEY).length > 0;
    known = MODULES.filter(function (m) { return !returning || ADDED_LATER.indexOf(m) < 0; });
    saveList(KNOWN_KEY, known);
  }
  if (lessonBtn) {
    var opened = loadList(OPENED_KEY), here = lessonBtn.getAttribute("data-module");
    if (opened.indexOf(here) < 0) { opened.push(here); saveList(OPENED_KEY, opened); }
  }

  function paint() {
    var done = load(), opened = loadList(OPENED_KEY);
    document.querySelectorAll("[data-module]").forEach(function (el) {
      var n = el.getAttribute("data-module"), isDone = done.indexOf(n) >= 0;
      el.classList.toggle("is-done", isDone);
      el.classList.toggle("is-started", !isDone && opened.indexOf(n) >= 0);
      // returning learners see a "new" marker on modules added since they started, until they open them
      el.classList.toggle("is-new", known.indexOf(n) < 0 && opened.indexOf(n) < 0 && !isDone);
    });
    var share = done.length / MODULES.length;
    document.querySelectorAll(".progress-bar").forEach(function (b) { b.style.width = (share * 100).toFixed(0) + "%"; });
    document.querySelectorAll(".progress-label").forEach(function (l) {
      l.textContent = done.length ? fmt("js_progress", { done: done.length, total: MODULES.length }, "{done} of {total} modules completed")
                                  : fmt("js_none", {}, "Not started yet");
    });
    var btn = document.querySelector(".complete-btn");
    if (btn) {
      var n = btn.getAttribute("data-module");
      btn.textContent = done.indexOf(n) >= 0 ? fmt("js_done", { n: n }, "✓ Module {n} completed (click to undo)")
                                             : fmt("js_mark", { n: n }, "Mark module {n} as complete");
    }
    var resume = document.querySelector(".resume");
    if (resume && done.length && done.length < MODULES.length) {
      var next = MODULES.find(function (m) { return done.indexOf(m) < 0; });
      var card = document.querySelector('.module-card[data-module="' + next + '"]');
      if (card) {
        var link = resume.querySelector(".resume-link");
        link.href = card.getAttribute("href");
        var vars = { n: next, title: card.querySelector("h3").textContent };
        link.textContent = card.classList.contains("is-new") ? fmt("js_resume_new", vars, "New since you started: module {n}, {title} →")
                                                             : fmt("js_resume", vars, "Continue with module {n}: {title} →");
        resume.hidden = false;
      }
    }
  }

  var btn = document.querySelector(".complete-btn");
  if (btn) {
    btn.addEventListener("click", function () {
      var n = btn.getAttribute("data-module"), done = load(), i = done.indexOf(n);
      if (i >= 0) done.splice(i, 1); else done.push(n);
      save(done); paint();
    });
  }

  // theme toggle: flips between light and dark, remembers the choice
  var toggle = document.querySelector(".theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var root = document.documentElement, cur = root.getAttribute("data-theme");
      if (!cur) cur = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
      var nextTheme = cur === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", nextTheme);
      try { localStorage.setItem("nn1001-theme", nextTheme); } catch (e) {}
    });
  }

  // mobile module list
  var side = document.querySelector(".sidebar");
  var sideToggle = document.querySelector(".side-toggle");
  if (side && sideToggle) {
    sideToggle.addEventListener("click", function () {
      var open = side.classList.toggle("open");
      sideToggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  // highlight the current section in the table of contents
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll(".toc a"));
  if (tocLinks.length && "IntersectionObserver" in window) {
    var byId = {};
    tocLinks.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting && byId[e.target.id]) {
          tocLinks.forEach(function (a) { a.classList.remove("active"); });
          byId[e.target.id].classList.add("active");
        }
      });
    }, { rootMargin: "-80px 0px -70% 0px" });
    document.querySelectorAll(".notebook h2[id]").forEach(function (h) { obs.observe(h); });
  }

  // hero animation: a small network whose activations ripple from left to right
  var svg = document.querySelector(".net");
  if (svg) {
    var NS = "http://www.w3.org/2000/svg";
    var layers = [3, 5, 5, 2], W = 420, H = 320, nodes = [];
    layers.forEach(function (n, li) {
      var x = 50 + li * ((W - 100) / (layers.length - 1)), col = [];
      for (var k = 0; k < n; k++) col.push({ x: x, y: H / 2 + (k - (n - 1) / 2) * 52 });
      nodes.push(col);
    });
    var edges = [], eg = svg.querySelector(".edges"), ng = svg.querySelector(".nodes");
    for (var li = 0; li < nodes.length - 1; li++) {
      nodes[li].forEach(function (a, ai) {
        nodes[li + 1].forEach(function (b, bi) {
          var l = document.createElementNS(NS, "line");
          l.setAttribute("x1", a.x); l.setAttribute("y1", a.y); l.setAttribute("x2", b.x); l.setAttribute("y2", b.y);
          eg.appendChild(l); edges.push({ el: l, layer: li });
        });
      });
    }
    var circles = nodes.map(function (col) {
      return col.map(function (p) {
        var c = document.createElementNS(NS, "circle");
        c.setAttribute("cx", p.x); c.setAttribute("cy", p.y); c.setAttribute("r", 11);
        ng.appendChild(c); return c;
      });
    });
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var step = 0;
    function tick() {
      var layer = step % (layers.length + 1);
      circles.forEach(function (col, li) {
        col.forEach(function (c) { c.classList.toggle("on", li === layer && Math.random() > 0.35); });
      });
      edges.forEach(function (e) { e.el.classList.toggle("pulse", e.layer === layer && Math.random() > 0.55); });
      step++;
    }
    tick();
    if (!reduce) setInterval(tick, 650);
  }

  paint();
})();
