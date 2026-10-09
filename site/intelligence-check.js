(function (root) {
  'use strict';

  function currentLang() {
    if (typeof root.AIFS_currentLang === 'function') return root.AIFS_currentLang();
    try {
      var fromQuery = new URLSearchParams(root.location.search).get('lang') || '';
      if (fromQuery) return fromQuery;
    } catch (_) {}
    try {
      return root.localStorage.getItem('lang') || 'en';
    } catch (_) {
      return 'en';
    }
  }

  function tHome(key, fallback) {
    var catalog = root.AIFS_UI_I18N || {};
    var lang = currentLang();
    var home = (catalog[lang] && catalog[lang].home) || {};
    var en = (catalog.en && catalog.en.home) || {};
    return home[key] || en[key] || fallback;
  }

  function repliesForLang() {
    return [
      [tHome('intelReply1a', 'Define “here”.'), tHome('intelReply1b', 'In the meantime, build something.')],
      [tHome('intelReply2a', 'Define “intelligence”.'), tHome('intelReply2b', 'A good place to start.')],
      [tHome('intelReply3a', 'Still worth learning backprop.'), tHome('intelReply3b', 'Some questions are best answered by building.')]
    ];
  }

  function applyStaticCopy(card) {
    if (!card) return;
    var eyebrow = card.querySelector('.intelligence-check-meta span:first-child');
    var question = card.querySelector('#intelligence-question, .intelligence-check-question');
    var label = card.querySelector('[data-intelligence-label]');
    var answer = card.querySelector('[data-intelligence-answer]');
    var hint = card.querySelector('[data-intelligence-hint]');
    var answered = card.getAttribute('data-state') === 'answered';

    if (eyebrow) eyebrow.textContent = tHome('intelEyebrow', 'Thought experiment');
    if (question) question.textContent = tHome('intelQuestion', 'Is superintelligence here yet?');
    if (label) {
      label.textContent = answered
        ? tHome('intelAskAgain', 'Ask again')
        : tHome('intelAsk', 'Ask the machine');
    }
    if (!answered) {
      if (answer) answer.textContent = tHome('intelUnofficial', 'A very unofficial test.');
      if (hint) hint.textContent = tHome('intelHint', 'Ask a big question. Get a small answer.');
    }
  }

  function init(doc, environment) {
    var card = doc && doc.querySelector('[data-intelligence-check]');
    if (!card) return false;

    if (card.getAttribute('data-intelligence-ready') === 'true') {
      applyStaticCopy(card);
      return false;
    }

    applyStaticCopy(card);

    var button = card.querySelector('[data-intelligence-trigger]');
    var answer = card.querySelector('[data-intelligence-answer]');
    var hint = card.querySelector('[data-intelligence-hint]');
    var label = card.querySelector('[data-intelligence-label]');
    var sponsor = card.querySelector('[data-intelligence-sponsor]');
    if (!button || !answer || !hint) return false;

    var runtime = environment || root;
    var count = 0;
    button.addEventListener('click', function (event) {
      var reduced = typeof runtime.matchMedia === 'function' && runtime.matchMedia('(prefers-reduced-motion: reduce)').matches;
      var replies = repliesForLang();
      var index = count % replies.length;
      count += 1;
      card.setAttribute('data-motion', event.detail === 0 || reduced ? 'instant' : 'animate');
      card.setAttribute('data-state', 'answered');
      card.setAttribute('data-check', String(count));
      card.setAttribute('data-dial', String(index + 1));
      answer.textContent = replies[index][0];
      hint.textContent = replies[index][1];
      if (label) label.textContent = tHome('intelAskAgain', 'Ask again');
      if (sponsor && count >= 2) sponsor.hidden = false;
    });

    card.setAttribute('data-intelligence-ready', 'true');
    button.disabled = false;
    return true;
  }

  function applyLanguage() {
    var card = root.document && root.document.querySelector('[data-intelligence-check]');
    if (!card) return;
    var check = Number(card.getAttribute('data-check') || '0');
    applyStaticCopy(card);
    if (card.getAttribute('data-state') === 'answered' && check > 0) {
      var replies = repliesForLang();
      var index = (check - 1) % replies.length;
      var answer = card.querySelector('[data-intelligence-answer]');
      var hint = card.querySelector('[data-intelligence-hint]');
      if (answer) answer.textContent = replies[index][0];
      if (hint) hint.textContent = replies[index][1];
    }
  }

  if (typeof module === 'object' && module.exports) {
    module.exports = { init: init, applyLanguage: applyLanguage };
  }
  if (root.document) {
    init(root.document, root);
    root.document.addEventListener('aifs:lang', applyLanguage);
    root.AIFS_applyIntelligenceCheckLang = applyLanguage;
  }
})(typeof window !== 'undefined' ? window : globalThis);
