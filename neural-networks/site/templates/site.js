(function () {
  var KEY = "nn1001-done";
  var MODULES = ["01", "02", "03", "04", "05", "06", "07", "08", "09"];

  function load() {
    try { return JSON.parse(localStorage.getItem(KEY) || "[]"); } catch (e) { return []; }
  }
  function save(list) {
    try { localStorage.setItem(KEY, JSON.stringify(list)); } catch (e) {}
  }

  function paint() {
    var done = load();
    document.querySelectorAll("[data-module]").forEach(function (el) {
      el.classList.toggle("is-done", done.indexOf(el.getAttribute("data-module")) >= 0);
    });
    var share = done.length / MODULES.length;
    document.querySelectorAll(".progress-bar").forEach(function (b) { b.style.width = (share * 100).toFixed(0) + "%"; });
    document.querySelectorAll(".progress-label").forEach(function (l) {
      l.textContent = done.length ? done.length + " of " + MODULES.length + " modules completed" : "Not started yet";
    });
    var btn = document.querySelector(".complete-btn");
    if (btn) {
      var n = btn.getAttribute("data-module");
      btn.textContent = done.indexOf(n) >= 0 ? "✓ Module " + n + " completed (click to undo)" : "Mark module " + n + " as complete";
    }
    var resume = document.querySelector(".resume");
    if (resume && done.length && done.length < MODULES.length) {
      var next = MODULES.find(function (m) { return done.indexOf(m) < 0; });
      var card = document.querySelector('.module-card[data-module="' + next + '"]');
      if (card) {
        var link = resume.querySelector(".resume-link");
        link.href = card.getAttribute("href");
        link.textContent = "Continue with module " + next + ": " + card.querySelector("h3").textContent + " →";
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
