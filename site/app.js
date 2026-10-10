(function () {
  var root = document.documentElement;
  var stored = '';
  try { stored = localStorage.getItem('theme') || ''; } catch (_) {}
  if (stored) {
    root.setAttribute('data-theme', stored);
  } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    root.setAttribute('data-theme', 'dark');
  } else {
    root.setAttribute('data-theme', 'light');
  }
  updateThemeIcon();

  document.addEventListener('DOMContentLoaded', function () {
    initThemeToggle();
    populateCurriculumSummary();
    populateStats();
    renderPhases();
    initStaggerIndex();
    initModal();
    initCopyButton();
    initMastheadFigure();
    initFadeObserver();
    initI18nEngine();
  });

  function populateCurriculumSummary() {
    if (typeof PHASES === 'undefined' || !Array.isArray(PHASES)) return;
    var lessonTotal = PHASES.reduce(function (total, phase) {
      return total + (Array.isArray(phase.lessons) ? phase.lessons.length : 0);
    }, 0);
    var lang = currentLang();
    var lessonLabel = lang === 'ko'
      ? formatCount(lessonTotal) + '개 레슨'
      : countLabel(lessonTotal, 'lesson', 'lessons');
    var phaseLabel = lang === 'ko'
      ? formatCount(PHASES.length) + '개 단계'
      : countLabel(PHASES.length, 'phase', 'phases');
    var values = {
      mastheadLessonCount: lessonLabel,
      mastheadPhaseCount: phaseLabel,
      prefaceLessonCount: lang === 'ko' ? formatCount(lessonTotal) + '개 공개 레슨' : lessonLabel,
      prefacePhaseCount: phaseLabel,
      tocLessonCount: lessonLabel,
      tocPhaseCount: phaseLabel
    };
    Object.keys(values).forEach(function (id) {
      var target = document.getElementById(id);
      if (target) target.textContent = values[id];
    });
  }

  function updateThemeIcon() {
    var icon = document.getElementById('themeIcon');
    if (!icon) return;
    var theme = root.getAttribute('data-theme');
    icon.textContent = theme === 'light' ? 'N' : 'D';
  }

  function initThemeToggle() {
    var btn = document.getElementById('themeToggle');
    if (!btn) return;
    btn.addEventListener('click', function () {
      var current = root.getAttribute('data-theme');
      var next = current === 'light' ? 'dark' : 'light';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (_) {}
      updateThemeIcon();
    });
    updateThemeIcon();
  }

  function computeStats() {
    var totalLessons = 0;
    var completeLessons = 0;
    var hasProgress = !!window.AIFSProgress;
    for (var i = 0; i < PHASES.length; i++) {
      var lessons = PHASES[i].lessons;
      totalLessons += lessons.length;
      for (var j = 0; j < lessons.length; j++) {
        var staticDone = lessons[j].status === 'complete';
        var userDone = false;
        if (hasProgress && lessons[j].url) {
          var lp = window.AIFSProgress.extractPath(lessons[j].url);
          if (lp) userDone = window.AIFSProgress.isLessonComplete(lp);
        }
        if (staticDone || userDone) completeLessons++;
      }
    }
    var completePhases = 0;
    for (var p = 0; p < PHASES.length; p++) {
      if (PHASES[p].status === 'complete') completePhases++;
    }
    return {
      lessons: totalLessons,
      phases: PHASES.length,
      complete: completeLessons,
      completePhases: completePhases
    };
  }

  function setBar(selector, pct) {
    var el = document.querySelector(selector);
    if (!el) return;
    var clamped = Math.max(0, Math.min(100, pct));
    el.setAttribute('data-target-pct', clamped.toFixed(1));
    if (el.classList.contains('in-view') || !window.IntersectionObserver) {
      setBarScale(el, clamped);
    } else {
      setBarScale(el, 0);
    }
  }

  function setBarScale(el, pct) {
    var clamped = Math.max(0, Math.min(100, Number(pct) || 0));
    el.style.setProperty('--bar-scale', (clamped / 100).toFixed(3));
  }

  function populateStats() {
    var stats = computeStats();
    var pct = stats.lessons > 0 ? (stats.complete / stats.lessons) * 100 : 0;
    var phasePct = stats.phases > 0 ? (stats.completePhases / stats.phases) * 100 : 0;
    var glossaryCount = (typeof GLOSSARY !== 'undefined') ? GLOSSARY.length : 0;

    setText('[data-stat="complete-frac"]', formatCount(stats.complete) + ' / ' + formatCount(stats.lessons));
    setText('[data-stat="phases-frac"]', formatCount(stats.completePhases) + ' / ' + formatCount(stats.phases));
    setText('[data-stat="glossary-count"]', formatCount(glossaryCount));
    setBar('[data-bar="complete"]', pct);
    setBar('[data-bar="phases"]', phasePct);
    setBar('[data-bar="languages"]', 100);
    setBar('[data-bar="glossary"]', glossaryCount > 0 ? 100 : 0);
  }

  function setText(selector, value) {
    var el = document.querySelector(selector);
    if (el) el.textContent = value;
  }

  function renderPhases() {
    var grid = document.getElementById('phasesGrid');
    if (!grid) return;
    if (!PHASES.length) {
      grid.innerHTML = '<p class="toc-empty">No phases are published yet.</p>';
      return;
    }
    var hasProgress = !!window.AIFSProgress;
    var html = '';
    for (var i = 0; i < PHASES.length; i++) {
      var p = PHASES[i];
      var total = p.lessons.length;
      var done = 0;
      for (var j = 0; j < p.lessons.length; j++) {
        var staticDone = p.lessons[j].status === 'complete';
        var userDone = false;
        if (hasProgress && p.lessons[j].url) {
          var lp = window.AIFSProgress.extractPath(p.lessons[j].url);
          if (lp) userDone = window.AIFSProgress.isLessonComplete(lp);
        }
        if (staticDone || userDone) done++;
      }
      var statusClass = escapeHtml(String(p.status || 'planned').replace(/ /g, '-'));
      var roman = toRoman(p.id);
      var num = String(p.id).padStart(2, '0');
      var phaseName = tPhaseName(p.id, p.name);
      html += '<div class="toc-row" data-phase="' + i + '" role="button" tabindex="0" aria-haspopup="dialog" aria-label="Open Phase ' + num + ': ' + escapeHtml(phaseName) + '">';
      html += '<span class="toc-num" dir="ltr">' + roman + '.</span>';
      html += '<div><span class="toc-status ' + statusClass + '"></span><span class="toc-name">' + escapeHtml(phaseName) + '</span></div>';
      html += '<span class="toc-meta">' + done + ' / ' + total + '</span>';
      html += '<span class="toc-meta">' + num + '</span>';
      html += '</div>';
    }
    grid.innerHTML = html;

    // Re-apply per-row stagger delays for the freshly created rows.
    initStaggerIndex();

    // If the reveal observer has already initialised (body.js-anim is set),
    // the IntersectionObserver is only watching the *original* rows it was
    // given at startup. Re-rendering via innerHTML replaces those nodes with
    // brand-new elements that are NOT being observed, so they would otherwise
    // stay hidden forever under `body.js-anim .toc-row { opacity: 0 }`.
    //
    // Since the user has already seen the initial reveal animation, just mark
    // the rebuilt rows as visible immediately (no second fade-in).
    if (document.body.classList.contains('js-anim')) {
      var newRows = grid.querySelectorAll('.toc-row');
      for (var r = 0; r < newRows.length; r++) {
        newRows[r].classList.add('in-view', 'visible');
      }
    }
  }

  function toRoman(num) {
    var lookup = [
      ['M', 1000], ['CM', 900], ['D', 500], ['CD', 400],
      ['C', 100], ['XC', 90], ['L', 50], ['XL', 40],
      ['X', 10], ['IX', 9], ['V', 5], ['IV', 4], ['I', 1]
    ];
    var n = parseInt(num, 10);
    if (isNaN(n) || n <= 0) return String(num);
    var out = '';
    for (var k = 0; k < lookup.length; k++) {
      while (n >= lookup[k][1]) {
        out += lookup[k][0];
        n -= lookup[k][1];
      }
    }
    return out;
  }

  function initModal() {
    var overlay = document.getElementById('modalOverlay');
    var modal = document.getElementById('modal');
    var closeBtn = document.getElementById('modalClose');
    if (!overlay || !modal || !closeBtn) return;

    overlay.setAttribute('aria-hidden', 'true');
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-modal', 'true');
    modal.setAttribute('aria-labelledby', 'modalTitle');
    modal.setAttribute('aria-describedby', 'modalDesc');
    closeBtn.setAttribute('aria-label', 'Close phase details');

    document.addEventListener('click', function (e) {
      var row = e.target.closest('.toc-row, .phase-card');
      if (row) {
        var idx = parseInt(row.getAttribute('data-phase'), 10);
        if (!isNaN(idx)) openModal(idx, false);
      }
    });

    document.addEventListener('keydown', function (e) {
      var row = e.target.closest && e.target.closest('.toc-row, .phase-card');
      if (!row || (e.key !== 'Enter' && e.key !== ' ')) return;
      e.preventDefault();
      var idx = parseInt(row.getAttribute('data-phase'), 10);
      if (!isNaN(idx)) openModal(idx, true);
    });

    closeBtn.addEventListener('click', function () { closeModal(false); });
    overlay.addEventListener('click', function (e) {
      if (e.target === overlay) closeModal(false);
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        closeModal(true);
        return;
      }
      if (e.key !== 'Tab' || !overlay.classList.contains('open')) return;
      var focusable = modal.querySelectorAll('a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])');
      if (!focusable.length) return;
      var first = focusable[0];
      var last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    });

    var resetBtn = document.getElementById('modalReset');
    if (resetBtn) {
      resetBtn.addEventListener('click', function () {
        if (!window.AIFSProgress) return;
        var ok = window.confirm(tHome('modalResetConfirm') || 'Clear all your local progress (quiz answers and completed lessons)? This cannot be undone.');
        if (!ok) return;
        window.AIFSProgress.reset();
      });
    }
  }

  var currentPhaseIdx = -1;
  var modalReturnFocus = null;

  function openModal(idx, fromKeyboard) {
    var p = PHASES[idx];
    if (!p) return;
    currentPhaseIdx = idx;
    modalReturnFocus = document.activeElement;

    var phaseName = tPhaseName(p.id, p.name);
    var phaseDesc = tPhaseDesc(p.id, p.desc);
    var lang = currentLang();
    document.getElementById('modalPhaseNum').textContent = (lang === 'ko' ? '단계 ' : 'PHASE ') + String(p.id).padStart(2, '0');
    document.getElementById('modalTitle').textContent = phaseName;
    document.getElementById('modalDesc').textContent = phaseDesc;

    renderModalLessons(p);

    var overlay = document.getElementById('modalOverlay');
    overlay.classList.toggle('no-motion', !!fromKeyboard);
    overlay.classList.add('open');
    overlay.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
    requestAnimationFrame(function () {
      var close = document.getElementById('modalClose');
      if (close) close.focus();
      overlay.classList.remove('no-motion');
    });
  }

  function renderModalLessons(p) {
    var container = document.getElementById('modalLessons');
    if (!container) return;

    var hasProgress = !!window.AIFSProgress;
    var userDone = 0;
    var html = '';

    for (var i = 0; i < p.lessons.length; i++) {
      var l = p.lessons[i];
      var pathMatch = l.url ? l.url.match(/(phases\/[^/]+\/[^/]+)\/?$/) : null;
      var lessonPath = pathMatch ? pathMatch[1] : '';
      var userComplete = hasProgress && lessonPath && window.AIFSProgress.isLessonComplete(lessonPath);
      if (userComplete) userDone++;

      var canOpen = (l.status === 'complete' || userComplete) && lessonPath;
      var lessonUrl = canOpen ? 'lesson?path=' + encodeURIComponent(lessonPath) : '';
      var tip = bilingualLessonTip(l.name, lessonPath);
      var lessonLabel = bilingualLessonNameHtml(l.name, lessonPath);
      var nameClass = 'modal-lesson-name' + (lessonTitleKo(lessonPath) ? ' has-ko' : '');
      var lessonMeta = '<span class="modal-lesson-meta"><span class="modal-lesson-type" data-type="' + escapeHtml(l.type) + '"' + (l.combines ? ' title="Combines: ' + escapeHtml(l.combines) + '"' : '') + '>' + escapeHtml(l.type) + '</span><span aria-hidden="true">·</span><span class="modal-lesson-lang">' + escapeHtml(l.lang) + '</span></span>';
      var lessonCopy = '<span class="modal-lesson-copy"><span class="' + nameClass + '" dir="auto" title="' + escapeHtml(tip) + '">' + lessonLabel + '</span>' + lessonMeta + '</span>';

      var openLabel = userComplete ? (tHome('reviewLesson') || 'Review') : (tHome('openLesson') || 'Open lesson');
      var comingSoonLabel = tHome('comingSoon') || 'Coming soon';
      var doneLabel = userComplete ? (tHome('done') || 'Done') : (tHome('markDone') || 'Mark done');
      var toggleTitle = userComplete ? (tHome('markNotDone') || 'Mark as not done') : (tHome('markDone') || 'Mark complete');

      html += '<div class="modal-lesson' + (userComplete ? ' user-done' : '') + '">';
      if (canOpen) {
        html += '<a href="' + lessonUrl + '" class="modal-lesson-open" title="' + escapeHtml(tip) + '" aria-label="' + escapeHtml(openLabel) + ': ' + escapeHtml(tip) + '">';
        html += lessonCopy;
        html += '<span class="modal-lesson-cta">' + escapeHtml(openLabel) + '<span aria-hidden="true">→</span></span></a>';
      } else {
        html += '<span class="modal-lesson-open is-unavailable" aria-disabled="true" title="' + escapeHtml(tip) + '">';
        html += lessonCopy;
        html += '<span class="modal-lesson-cta">' + escapeHtml(comingSoonLabel) + '</span></span>';
      }

      var toggleHtml = '';
      if (hasProgress && canOpen) {
        toggleHtml = '<button type="button" class="modal-lesson-toggle' + (userComplete ? ' done' : '') + '" data-path="' + lessonPath + '" title="' + escapeHtml(toggleTitle) + '" aria-label="' + escapeHtml(toggleTitle) + '"><span class="modal-lesson-check" aria-hidden="true">' + (userComplete ? '✓' : '') + '</span><span class="modal-lesson-toggle-label">' + escapeHtml(doneLabel) + '</span></button>';
      }
      html += toggleHtml;
      html += '</div>';
    }

    container.innerHTML = html || '<p class="modal-lessons-empty">No lessons are published in this phase yet.</p>';

    var toggles = container.querySelectorAll('.modal-lesson-toggle');
    for (var t = 0; t < toggles.length; t++) {
      toggles[t].addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        var path = this.getAttribute('data-path');
        if (!path || !window.AIFSProgress) return;
        if (window.AIFSProgress.isLessonComplete(path)) {
          window.AIFSProgress.unmarkLessonComplete(path);
        } else {
          window.AIFSProgress.markLessonComplete(path);
        }
      });
    }

    var progEl = document.getElementById('modalProgress');
    var barEl = document.getElementById('modalProgressBar');
    var barFill = document.getElementById('modalProgressBarFill');
    if (hasProgress && p.lessons.length > 0) {
      var pct = Math.round((userDone / p.lessons.length) * 100);
      if (progEl) {
        progEl.style.display = '';
        progEl.innerHTML = '<span><strong class="modal-progress-count">' + formatCount(userDone) + '</strong> of ' + countLabel(p.lessons.length, 'lesson', 'lessons') + ' complete</span><span class="modal-progress-pct">' + pct + '%</span>';
      }
      if (barEl && barFill) {
        barEl.style.display = '';
        barEl.setAttribute('role', 'progressbar');
        barEl.setAttribute('aria-label', p.name + ' progress');
        barEl.setAttribute('aria-valuemin', '0');
        barEl.setAttribute('aria-valuemax', '100');
        barEl.setAttribute('aria-valuenow', String(pct));
        barFill.style.transform = 'scaleX(' + (pct / 100) + ')';
      }
    } else {
      if (progEl) progEl.style.display = 'none';
      if (barEl) barEl.style.display = 'none';
    }
  }

  if (window.AIFSProgress) {
    window.AIFSProgress.onChange(function () {
      if (currentPhaseIdx >= 0 && PHASES[currentPhaseIdx]) {
        renderModalLessons(PHASES[currentPhaseIdx]);
      }
      populateStats();
      renderPhases();
    });
  }

  function closeModal(fromKeyboard) {
    var overlay = document.getElementById('modalOverlay');
    if (!overlay || !overlay.classList.contains('open')) return;
    overlay.classList.toggle('no-motion', !!fromKeyboard);
    overlay.classList.remove('open');
    overlay.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
    if (modalReturnFocus && modalReturnFocus.isConnected && typeof modalReturnFocus.focus === 'function') {
      modalReturnFocus.focus();
    }
    modalReturnFocus = null;
    requestAnimationFrame(function () {
      overlay.classList.remove('no-motion');
    });
  }

  // One clipboard implementation for every copy chip on the site: debounced
  // copied-state revert, execCommand fallback when the async API is denied.
  function wireCopyButton(btn, label, getText) {
    if (!btn || !label) return;
    var revertTimer = null;
    var defaultLabel = label.textContent || 'copy';
    var defaultAriaLabel = btn.getAttribute('aria-label') || 'Copy command';
    function resetCopyState() {
      label.textContent = defaultLabel;
      btn.classList.remove('copied');
      btn.setAttribute('aria-label', defaultAriaLabel);
    }
    function scheduleReset() {
      if (revertTimer) clearTimeout(revertTimer);
      revertTimer = setTimeout(resetCopyState, 1500);
    }
    function confirmCopied() {
      label.textContent = 'copied';
      btn.classList.add('copied');
      btn.setAttribute('aria-label', 'Command copied');
      scheduleReset();
    }
    function reportCopyFailure() {
      label.textContent = 'retry';
      btn.classList.remove('copied');
      btn.setAttribute('aria-label', 'Copy failed. Try again');
      scheduleReset();
    }
    function fallbackCopy(text) {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.top = '0';
      ta.style.left = '0';
      ta.style.width = '1px';
      ta.style.height = '1px';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      ta.setSelectionRange(0, ta.value.length);
      var copied = false;
      try { copied = document.execCommand('copy'); } catch (e) {}
      ta.remove();
      if (copied) confirmCopied();
      else reportCopyFailure();
    }
    btn.addEventListener('click', function () {
      var text = getText();
      if (!text) {
        reportCopyFailure();
        return;
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(confirmCopied).catch(function () { fallbackCopy(text); });
      } else {
        fallbackCopy(text);
      }
    });
  }

  function initCopyButton() {
    var code = document.getElementById('cloneCmd');
    if (code) {
      wireCopyButton(
        document.getElementById('copyBtn'),
        document.getElementById('copyBtnLabel'),
        function () { return code.textContent; }
      );
    }
    var installBtn = document.getElementById('installCopy');
    if (installBtn) {
      wireCopyButton(
        installBtn,
        document.getElementById('installCopyLabel'),
        function () { return installBtn.getAttribute('data-cmd'); }
      );
    }
  }

  function initMastheadFigure() {
    var figure = document.querySelector('[data-masthead-figure]');
    if (!figure) return;
    var panels = Array.prototype.slice.call(figure.querySelectorAll('.fig-panel'));
    var dots = Array.prototype.slice.call(figure.querySelectorAll('.fig-dot'));
    var previous = figure.querySelector('.fig-previous');
    var next = figure.querySelector('.fig-next');
    var controls = figure.querySelector('.fig-controls');
    var caption = figure.querySelector('.fig-caption');
    if (panels.length < 2 || dots.length !== panels.length || !previous || !next || !controls || !caption) return;

    var autoplayDelay = 6500;
    var current = Math.max(0, panels.findIndex(function (panel) { return panel.classList.contains('is-active'); }));
    var reducedQuery = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
    var desktopQuery = window.matchMedia ? window.matchMedia('(min-width: 1280px) and (hover: hover) and (pointer: fine)') : null;
    var inViewport = !window.IntersectionObserver;
    var timer = 0;
    var timerStartedAt = 0;
    var timerRemaining = autoplayDelay;
    var autoplayCancelled = !!(reducedQuery && reducedQuery.matches);
    var autoplayComplete = false;
    var disposed = false;
    var cleanups = [];

    function now() {
      return window.performance && typeof window.performance.now === 'function' ? window.performance.now() : Date.now();
    }

    function listen(target, type, handler, options) {
      target.addEventListener(type, handler, options);
      cleanups.push(function () { target.removeEventListener(type, handler, options); });
    }

    function listenToQuery(query, handler) {
      if (!query) return;
      if (typeof query.addEventListener === 'function') {
        query.addEventListener('change', handler);
        cleanups.push(function () { query.removeEventListener('change', handler); });
      } else if (typeof query.addListener === 'function') {
        query.addListener(handler);
        cleanups.push(function () { query.removeListener(handler); });
      }
    }

    function isDesktopView() {
      return desktopQuery ? desktopQuery.matches : figure.getClientRects().length > 0;
    }

    function isReduced() {
      return !!(reducedQuery && reducedQuery.matches);
    }

    function isOnScreen() {
      var rect = figure.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.top < window.innerHeight && rect.right > 0 && rect.left < window.innerWidth;
    }

    function showPlate(index, announce) {
      current = Math.max(0, Math.min(panels.length - 1, index));
      panels.forEach(function (panel, panelIndex) {
        var active = panelIndex === current;
        panel.classList.toggle('is-active', active);
        panel.setAttribute('aria-hidden', active ? 'false' : 'true');
      });
      dots.forEach(function (dot, dotIndex) {
        var active = dotIndex === current;
        dot.classList.toggle('is-active', active);
        dot.setAttribute('aria-pressed', active ? 'true' : 'false');
      });
      caption.setAttribute('aria-live', announce ? 'polite' : 'off');
      var plateKey = 'plateCaption' + (current + 1);
      var localizedCaption = tHome(plateKey);
      caption.textContent = localizedCaption || ('Plate ' + (current + 1) + ' of ' + panels.length + '. ' + panels[current].getAttribute('data-caption'));
      previous.textContent = tHome('figPrevious') || 'Previous';
      next.textContent = tHome('figNext') || 'Next';
      previous.disabled = current === 0;
      next.disabled = current === panels.length - 1;
    }

    function clearTimer(preserveRemaining) {
      if (!timer) return;
      if (preserveRemaining) {
        timerRemaining = Math.max(0, timerRemaining - (now() - timerStartedAt));
      }
      window.clearTimeout(timer);
      timer = 0;
      timerStartedAt = 0;
    }

    function canAutoplay() {
      return !autoplayCancelled && !autoplayComplete && !isReduced() && isDesktopView() && inViewport && !document.hidden && figure.getClientRects().length > 0;
    }

    function scheduleAutoplay() {
      if (timer || !canAutoplay()) return;
      if (current >= panels.length - 1) {
        autoplayComplete = true;
        figure.removeAttribute('data-autoplay');
        return;
      }
      figure.setAttribute('data-autoplay', 'true');
      timerStartedAt = now();
      timer = window.setTimeout(function () {
        timer = 0;
        timerStartedAt = 0;
        showPlate(current + 1, false);
        timerRemaining = autoplayDelay;
        if (current >= panels.length - 1) {
          autoplayComplete = true;
          figure.removeAttribute('data-autoplay');
          return;
        }
        scheduleAutoplay();
      }, Math.max(16, timerRemaining));
    }

    function cancelAutoplay() {
      clearTimer(false);
      autoplayCancelled = true;
      figure.removeAttribute('data-autoplay');
    }

    function syncRuntime() {
      if (disposed) return;
      var paused = isReduced() || !isDesktopView() || !inViewport || document.hidden || figure.getClientRects().length === 0;
      figure.setAttribute('data-motion-paused', paused ? 'true' : 'false');
      if (paused) clearTimer(true);
      else scheduleAutoplay();
    }

    function choosePlate(index) {
      cancelAutoplay();
      showPlate(index, true);
      syncRuntime();
    }

    dots.forEach(function (dot, index) {
      listen(dot, 'click', function () { choosePlate(index); });
    });
    listen(previous, 'click', function () { choosePlate(current - 1); });
    listen(next, 'click', function () { choosePlate(current + 1); });
    listen(controls, 'pointerdown', cancelAutoplay);
    listen(controls, 'keydown', cancelAutoplay);
    listen(controls, 'focusin', cancelAutoplay);
    listen(document, 'visibilitychange', syncRuntime);

    listenToQuery(reducedQuery, function () {
      if (isReduced()) cancelAutoplay();
      syncRuntime();
    });
    listenToQuery(desktopQuery, function () {
      inViewport = isOnScreen();
      syncRuntime();
    });

    if (window.IntersectionObserver) {
      var observer = new IntersectionObserver(function (entries) {
        if (!entries.length) return;
        inViewport = entries[0].isIntersecting && entries[0].intersectionRatio > 0;
        syncRuntime();
      }, { threshold: 0.12 });
      observer.observe(figure);
      cleanups.push(function () { observer.disconnect(); });
    } else {
      var checkViewport = function () {
        inViewport = isOnScreen();
        syncRuntime();
      };
      listen(window, 'scroll', checkViewport, { passive: true });
      listen(window, 'resize', checkViewport);
      checkViewport();
    }

    function cleanup() {
      if (disposed) return;
      disposed = true;
      clearTimer(false);
      while (cleanups.length) cleanups.pop()();
      figure.removeAttribute('data-autoplay');
      figure.setAttribute('data-motion-paused', 'true');
    }

    listen(window, 'pagehide', function (event) {
      if (!event.persisted) cleanup();
    });
    showPlate(current, false);
    syncRuntime();
  }

  function initFadeObserver() {
    var prefersReduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var visibleEls = document.querySelectorAll('.reveal, .fade-in, .ascii-rule, .toc-row');
    for (var v = 0; v < visibleEls.length; v++) {
      visibleEls[v].classList.add('in-view', 'visible');
    }

    if (!window.IntersectionObserver || prefersReduced) {
      document.querySelectorAll('.stat-row-bar').forEach(function (el) {
        el.classList.add('in-view', 'visible');
        var target = el.getAttribute('data-target-pct');
        if (target !== null) setBarScale(el, target);
      });
      return;
    }

    var els = document.querySelectorAll('.stat-row-bar');
    if (!els.length) return;
    var observer = new IntersectionObserver(function (entries) {
      for (var i = 0; i < entries.length; i++) {
        if (entries[i].isIntersecting) {
          var el = entries[i].target;
          el.classList.add('in-view', 'visible');
          var target = el.getAttribute('data-target-pct');
          if (target !== null) {
            setBarScale(el, target);
          }
          observer.unobserve(el);
        }
      }
    }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
    for (var i = 0; i < els.length; i++) {
      observer.observe(els[i]);
    }
  }

  function initStaggerIndex() {
    var rows = document.querySelectorAll('.toc-list .toc-row');
    for (var i = 0; i < rows.length; i++) {
      rows[i].style.setProperty('--stagger-delay', (i * 30) + 'ms');
    }
  }

  function formatCount(value) {
    return (Number(value) || 0).toLocaleString('en');
  }

  function countLabel(value, singular, plural) {
    return formatCount(value) + ' ' + (Number(value) === 1 ? singular : plural);
  }

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str == null ? '' : str;
    return div.innerHTML.replace(/"/g, '&quot;');
  }

  const currentLang = () => {
    if (window.AIFS_currentLang && typeof window.AIFS_currentLang === 'function') {
      return window.AIFS_currentLang();
    }
    return 'en';
  };

  const getI18nDict = () => {
    var lang = currentLang();
    var catalog = window.AIFS_UI_I18N || {};
    return catalog[lang] || catalog.en || {};
  };

  const tHome = (key) => {
    var dict = getI18nDict();
    var home = dict.home || {};
    var fallback = (window.AIFS_UI_I18N && window.AIFS_UI_I18N.en && window.AIFS_UI_I18N.en.home) || {};
    return home[key] || fallback[key] || '';
  };

  const tRoute = (key) => {
    var dict = getI18nDict();
    var routes = dict.routes || {};
    var fallback = (window.AIFS_UI_I18N && window.AIFS_UI_I18N.en && window.AIFS_UI_I18N.en.routes) || {};
    return routes[key] || fallback[key] || '';
  };

  const tPhaseName = (id, defaultName) => {
    var dict = getI18nDict();
    var names = dict.phaseNames || {};
    return names[id] || defaultName;
  };

  const tPhaseDesc = (id, defaultDesc) => {
    var dict = getI18nDict();
    var descs = dict.phaseDescs || {};
    return descs[id] || defaultDesc;
  };

  function lessonTitleKo(path) {
    var map = window.AIFS_LESSON_TITLES_KO || {};
    return path && map[path] ? map[path] : '';
  }

  function bilingualLessonNameHtml(name, path) {
    var ko = lessonTitleKo(path);
    var html = escapeHtml(name);
    if (ko) html += '<span class="lesson-title-ko">' + escapeHtml(ko) + '</span>';
    return html;
  }

  function bilingualLessonTip(name, path) {
    var ko = lessonTitleKo(path);
    return ko ? (name + ' · ' + ko) : name;
  }

  const applyPageTranslations = (lang) => {
    var isKorean = (lang === 'ko');

    // 1. Header navigation
    var navMap = {
      '#contents': tHome('navContents') || 'Contents',
      '#books': tHome('navBooks') || 'Books',
      'catalog.html': tHome('navCatalog') || 'Catalog',
      'prereqs.html': tHome('navRoadmap') || 'Roadmap',
      'glossary.html': tHome('navGlossary') || 'Glossary',
      'about.html': tHome('navAbout') || 'About',
      'certifications.html': tHome('navCertifications') || 'Certifications',
      'sponsors.html': tHome('navSponsor') || 'Sponsor us'
    };
    document.querySelectorAll('.header-nav a').forEach(function (link) {
      var href = link.getAttribute('href');
      if (link.hasAttribute('data-newsletter-open')) {
        link.textContent = tHome('navNewsletter') || 'Newsletter';
        return;
      }
      if (href && navMap[href]) link.textContent = navMap[href];
    });

    // 2. Masthead
    var metaRow = document.querySelector('.manual-meta-row .right');
    if (metaRow) metaRow.textContent = tHome('openSourceMit') || 'open source · MIT';

    var tagline = document.querySelector('.manual-tagline');
    if (tagline) {
      var suffix = tHome('taglineSuffix') || 'Every algorithm built from raw math before a single framework gets imported.';
      tagline.innerHTML = '<span id="mastheadLessonCount"></span>. <span id="mastheadPhaseCount"></span>. ' + escapeHtml(suffix);
    }

    var attribution = document.querySelector('.manual-attribution');
    if (attribution) attribution.textContent = tHome('attribution') || 'Maintained by Rohit Ghumare and contributors. Run on your own machine.';

    var ctaPrimary = document.querySelector('.masthead-btn--primary span');
    if (ctaPrimary) ctaPrimary.textContent = tHome('btnStartCourse') || 'Start the Course';

    var ctaPaths = document.querySelector('.masthead-btn[href="learning-paths.html"] span');
    if (ctaPaths) ctaPaths.textContent = tHome('btnExplorePaths') || 'Explore Learning Paths';

    var ctaStar = document.querySelector('a[aria-label="Star ai-engineering-from-scratch on GitHub"] span:not(.masthead-btn-count)');
    if (ctaStar) ctaStar.textContent = tHome('btnStarGitHub') || 'Star on GitHub';

    var ctaFollow = document.querySelector('a[aria-label="Follow Rohit Ghumare on GitHub"] span');
    if (ctaFollow) ctaFollow.textContent = tHome('btnFollow') || 'Follow @rohitg00';

    var installTitle = document.querySelector('.masthead-install-bar > span');
    if (installTitle) installTitle.textContent = tHome('terminalLearn') || 'Learn in your terminal';

    var installCopyLabel = document.getElementById('installCopyLabel');
    if (installCopyLabel) installCopyLabel.textContent = tHome('terminalCopy') || 'copy';

    var installCaption = document.querySelector('.masthead-install-caption');
    if (installCaption) installCaption.textContent = tHome('terminalCaption') || 'Your agent becomes your tutor: placement quiz, personalized path, lessons taught interactively in your terminal.';

    var figPrev = document.querySelector('.fig-previous');
    if (figPrev) figPrev.textContent = tHome('figPrevious') || 'Previous';
    var figNext = document.querySelector('.fig-next');
    if (figNext) figNext.textContent = tHome('figNext') || 'Next';

    // 3. Learners strip
    var learnersEyebrow = document.querySelector('.learners-eyebrow');
    if (learnersEyebrow) learnersEyebrow.textContent = tHome('learnersEyebrow') || 'Read by engineers and students at';

    var learnersQuote = document.querySelector('.learners-quote');
    if (learnersQuote) learnersQuote.innerHTML = tHome('learnersQuote') || '&ldquo;Obsessed with the AI Engineering from Scratch repo.&rdquo; <span class="learners-quote-attr">- AI engineer at Google</span>';

    // 4. Preface
    var prefaceEyebrow = document.querySelector('.preface-eyebrow');
    if (prefaceEyebrow) prefaceEyebrow.textContent = tHome('prefaceEyebrow') || 'How this works';

    var prefaceBody = document.querySelector('.preface-body');
    if (prefaceBody) {
      var paragraphs = prefaceBody.querySelectorAll('p');
      if (paragraphs.length >= 3) {
        paragraphs[0].textContent = tHome('prefaceP1') || paragraphs[0].textContent;
        var p2Prefix = tHome('prefaceP2Prefix') || 'This curriculum is the spine.';
        var p2Suffix = tHome('prefaceP2Suffix') || 'four languages: Python, TypeScript, Rust, Julia...';
        paragraphs[1].innerHTML = p2Prefix + ' <span id="prefacePhaseCount"></span>, <span id="prefaceLessonCount"></span>, ' + p2Suffix;
        paragraphs[2].textContent = tHome('prefaceP3') || paragraphs[2].textContent;
      }
    }

    // 5. Course paths
    var coursePathsTitle = document.getElementById('coursePathsTitle');
    if (coursePathsTitle) coursePathsTitle.textContent = tHome('coursePathsTitle') || 'Choose the work you want to do';

    var coursePathsCopy = document.querySelector('.course-paths-header-copy > p');
    if (coursePathsCopy) coursePathsCopy.textContent = tHome('coursePathsHeaderCopy') || 'AI engineering is larger than model code...';

    var viewPathsLink = document.querySelector('.course-paths-entry-links a[href="learning-paths.html"]');
    if (viewPathsLink) viewPathsLink.textContent = tHome('viewLearningPaths') || 'View Learning Paths';

    var browseRoutesLink = document.querySelector('.course-paths-entry-links a[href="learning-paths.html#career-routes"]');
    if (browseRoutesLink) browseRoutesLink.textContent = tHome('browseCareerRoutes') || 'Browse career routes';

    var lpRoot = document.querySelector('.learning-paths-compact-root strong');
    if (lpRoot) lpRoot.textContent = tHome('learningPathsRoot') || 'AI Engineering';
    var lpSub = document.querySelector('.learning-paths-compact-root span');
    if (lpSub) lpSub.textContent = tHome('learningPathsSub') || '4 connected domains';

    var appLabel = document.querySelector('.learning-paths-node--applications .learning-paths-node-label');
    if (appLabel) appLabel.textContent = tHome('pathAppLabel') || 'Building and Deploying AI Applications';
    var softwareLabel = document.querySelector('.learning-paths-node--software .learning-paths-node-label');
    if (softwareLabel) softwareLabel.textContent = tHome('pathSoftwareLabel') || 'Software Engineering Fundamentals';
    var agentsLabel = document.querySelector('.learning-paths-node--agents .learning-paths-node-label');
    if (agentsLabel) agentsLabel.textContent = tHome('pathAgentsLabel') || 'Agent-Assisted Engineering';
    var shapingLabel = document.querySelector('.learning-paths-node--shaping .learning-paths-node-label');
    if (shapingLabel) shapingLabel.textContent = tHome('pathShapingLabel') || 'Product Judgment and Delivery';

    // Route cards
    var routeCards = document.querySelectorAll('.course-route');
    routeCards.forEach(function (card) {
      var nameEl = card.querySelector('.course-route-name strong');
      var tagEl = card.querySelector('.course-route-name span');
      var descEl = card.querySelector('p');
      var links = card.querySelectorAll('.course-route-actions a');
      if (!nameEl) return;
      var routeId = card.getAttribute('data-route-id');
      if (!routeId) {
        var txt = nameEl.textContent.trim().toLowerCase();
        if (txt.indexOf('new to') >= 0) routeId = 'newToAi';
        else if (txt.indexOf('building') >= 0) routeId = 'app';
        else if (txt.indexOf('software') >= 0) routeId = 'software';
        else if (txt.indexOf('agent-assisted') >= 0) routeId = 'agents';
        else if (txt.indexOf('product') >= 0) routeId = 'shaping';
        else if (txt.indexOf('model context') >= 0) routeId = 'mcp';
        else if (txt.indexOf('agent skills') >= 0) routeId = 'skills';
        else if (txt.indexOf('certification') >= 0) routeId = 'cert';
        card.setAttribute('data-route-id', routeId || '');
      }

      if (routeId === 'newToAi') {
        if (tagEl) tagEl.textContent = tRoute('recFirst') || 'Recommended first';
        nameEl.textContent = tRoute('newToAiTitle') || 'New to AI engineering';
        if (descEl) descEl.textContent = tRoute('newToAiDesc') || descEl.textContent;
        if (links[0]) links[0].textContent = tRoute('openLesson') || 'Open lesson';
        if (links[1]) links[1].textContent = tRoute('ghSource') || 'GitHub source';
      } else if (routeId === 'app') {
        if (tagEl) tagEl.textContent = tRoute('coreDomain') || 'Core domain';
        nameEl.textContent = tRoute('appTitle') || 'Building and Deploying AI Applications';
        if (descEl) descEl.textContent = tRoute('appDesc') || descEl.textContent;
        if (links[0]) links[0].textContent = tRoute('startPath') || 'Start path';
        if (links[1]) links[1].textContent = tRoute('ghPath') || 'GitHub path';
      } else if (routeId === 'software') {
        if (tagEl) tagEl.textContent = tRoute('coreDomain') || 'Core domain';
        nameEl.textContent = tRoute('softwareTitle') || 'Software Engineering Fundamentals';
        if (descEl) descEl.textContent = tRoute('softwareDesc') || descEl.textContent;
        if (links[0]) links[0].textContent = tRoute('startPath') || 'Start path';
        if (links[1]) links[1].textContent = tRoute('ghPath') || 'GitHub path';
      } else if (routeId === 'agents') {
        if (tagEl) tagEl.textContent = tRoute('coreDomain') || 'Core domain';
        nameEl.textContent = tRoute('agentsTitle') || 'Agent-Assisted Engineering';
        if (descEl) descEl.textContent = tRoute('agentsDesc') || descEl.textContent;
        if (links[0]) links[0].textContent = tRoute('startPath') || 'Start path';
        if (links[1]) links[1].textContent = tRoute('ghPath') || 'GitHub path';
      } else if (routeId === 'shaping') {
        if (tagEl) tagEl.textContent = tRoute('coreDomain') || 'Core domain';
        nameEl.textContent = tRoute('shapingTitle') || 'Product Judgment and Delivery';
        if (descEl) descEl.textContent = tRoute('shapingDesc') || descEl.textContent;
        if (links[0]) links[0].textContent = tRoute('startPath') || 'Start path';
        if (links[1]) links[1].textContent = tRoute('ghPath') || 'GitHub path';
      } else if (routeId === 'mcp') {
        if (tagEl) tagEl.textContent = tRoute('focusedPath') || 'Focused path';
        nameEl.textContent = tRoute('mcpTitle') || 'Model Context Protocol (MCP)';
        if (descEl) descEl.textContent = tRoute('mcpDesc') || descEl.textContent;
        if (links[0]) links[0].textContent = tRoute('startPath') || 'Start path';
        if (links[1]) links[1].textContent = tRoute('ghSource') || 'GitHub source';
      } else if (routeId === 'skills') {
        if (tagEl) tagEl.textContent = tRoute('focusedPath') || 'Focused path';
        nameEl.textContent = tRoute('skillsTitle') || 'Agent Skills';
        if (descEl) descEl.textContent = tRoute('skillsDesc') || descEl.textContent;
        if (links[0]) links[0].textContent = tRoute('startPath') || 'Start path';
        if (links[1]) links[1].textContent = tRoute('ghSource') || 'GitHub source';
      } else if (routeId === 'cert') {
        if (tagEl) tagEl.textContent = tRoute('practiceEvidence') || 'Practice by evidence';
        nameEl.textContent = tRoute('certTitle') || 'Certification preparation';
        if (descEl) descEl.textContent = tRoute('certDesc') || descEl.textContent;
        if (links[0]) links[0].textContent = tRoute('explorePaths') || 'Explore paths';
        if (links[1]) links[1].textContent = tRoute('ghTutor') || 'GitHub tutor';
      }
    });

    // 6. Stats block
    var statTitle = document.querySelector('.stat-block-title');
    if (statTitle) statTitle.textContent = tHome('statBlockTitle') || 'Current Progress';

    var statLabels = document.querySelectorAll('.stat-row-label');
    if (statLabels.length >= 4) {
      statLabels[0].textContent = tHome('statFinishedLessons') || 'Finished Lessons';
      statLabels[1].textContent = tHome('statPhases') || 'Phases';
      statLabels[2].textContent = tHome('statLanguages') || 'Languages';
      statLabels[3].textContent = tHome('statGlossaryTerms') || 'Glossary Terms';
    }

    // 7. Contents (TOC)
    var tocTitle = document.querySelector('#contents .toc-title');
    if (tocTitle && typeof PHASES !== 'undefined') {
      tocTitle.textContent = isKorean
        ? (tHome('tocTitleSuffix') || '전체 커리큘럼') + ' · ' + PHASES.length + '개 단계 · 523개 레슨'
        : 'Curriculum · ' + PHASES.length + ' phases · 523 lessons';
    }
    var tocSubtitle = document.querySelector('#contents .toc-subtitle');
    if (tocSubtitle) tocSubtitle.textContent = tHome('tocSubtitle') || 'Tap a phase to expand its lessons...';

    var legendItems = document.querySelectorAll('.legend .legend-item');
    if (legendItems.length >= 3) {
      legendItems[0].innerHTML = '<span class="toc-status complete"></span> ' + (tHome('legendComplete') || 'Complete');
      legendItems[1].innerHTML = '<span class="toc-status in-progress"></span> ' + (tHome('legendInProgress') || 'In progress');
      legendItems[2].innerHTML = '<span class="toc-status planned"></span> ' + (tHome('legendPlanned') || 'Planned');
    }

    // 8. Books, Colophon, Footer
    var booksTitle = document.querySelector('#books .toc-title');
    if (booksTitle) booksTitle.textContent = tHome('booksTitle') || 'The book edition · six volumes';
    var booksSubtitle = document.querySelector('#books .toc-subtitle');
    if (booksSubtitle) booksSubtitle.textContent = tHome('booksSubtitle') || 'The course, compiled...';
    var booksNote = document.querySelector('.books-note');
    if (booksNote) booksNote.innerHTML = tHome('booksNote') || booksNote.innerHTML;

    var colophonEyebrow = document.querySelector('.colophon-eyebrow');
    if (colophonEyebrow) colophonEyebrow.textContent = tHome('colophonEyebrow') || 'Colophon';
    var colophonP = document.querySelector('.colophon-grid .reveal p');
    if (colophonP) colophonP.textContent = tHome('colophonText') || colophonP.textContent;
    var copyBtnLabel = document.getElementById('copyBtnLabel');
    if (copyBtnLabel) copyBtnLabel.textContent = tHome('terminalCopy') || 'copy';

    var footerP = document.querySelector('.site-footer p');
    if (footerP) footerP.textContent = tHome('footerCopy') || '© 2026 · open source · free forever';
    var footerMap = {
      'about.html': tHome('navAbout') || 'About',
      'certifications.html': tHome('navCertifications') || 'Certifications',
      'catalog.html': tHome('navCatalog') || 'Catalog',
      'glossary.html': tHome('navGlossary') || 'Glossary',
      'sponsors.html': tHome('navSponsor') || 'Sponsor us'
    };
    document.querySelectorAll('.footer-links a').forEach(function (link) {
      var href = link.getAttribute('href') || '';
      if (link.hasAttribute('data-newsletter-open')) {
        link.textContent = tHome('navNewsletter') || 'Newsletter';
        return;
      }
      if (footerMap[href]) link.textContent = footerMap[href];
      else if (href.indexOf('issues') >= 0) link.textContent = tHome('navReport') || 'Report';
    });

    if (typeof window.AIFS_applyIntelligenceCheckLang === 'function') {
      window.AIFS_applyIntelligenceCheckLang();
    }
    if (typeof window.AIFS_applyNewsletterLang === 'function') {
      window.AIFS_applyNewsletterLang();
    }

    populateCurriculumSummary();
    renderPhases();
    var modalOverlay = document.getElementById('modalOverlay');
    if (modalOverlay && modalOverlay.classList.contains('open') && currentPhaseIdx >= 0) {
      openModal(currentPhaseIdx, false);
    }
  };

  const handleLanguageChange = (newLang) => {
    applyPageTranslations(newLang);
  };

  const initI18nEngine = () => {
    var lang = currentLang();
    applyPageTranslations(lang);
    window.AIFS_onLangChange = handleLanguageChange;
  };
})();
