(function () {
  'use strict';

  const publicationUrl = 'https://thatdevopsguy.substack.com';
  let dialog;
  let trigger;

  function currentLang() {
    if (typeof window.AIFS_currentLang === 'function') return window.AIFS_currentLang();
    try {
      const fromQuery = new URLSearchParams(location.search).get('lang') || '';
      if (fromQuery) return fromQuery;
    } catch (_) {}
    try {
      return localStorage.getItem('lang') || 'en';
    } catch (_) {
      return 'en';
    }
  }

  function tHome(key, fallback) {
    const catalog = window.AIFS_UI_I18N || {};
    const lang = currentLang();
    const home = (catalog[lang] && catalog[lang].home) || {};
    const en = (catalog.en && catalog.en.home) || {};
    return home[key] || en[key] || fallback;
  }

  function signupSection(variant) {
    const lesson = variant === 'lesson';
    const id = variant === 'dialog' ? 'newsletter-modal' : 'newsletter';
    const section = document.createElement('section');
    section.id = id;
    section.className = 'newsletter-section newsletter-' + variant;
    if (variant === 'home') section.classList.add('container');
    section.setAttribute('aria-labelledby', id + '-title');
    section.dataset.newsletterVariant = variant;
    const title = lesson
      ? tHome('newsletterLessonTitle', 'AI Engineering Newsletter')
      : tHome('newsletterHomeTitle', 'Learn AI from scratch. Stay ahead of what’s next.');
    const body = lesson
      ? tHome('newsletterLessonBody', 'Practical lessons and updates on AI, DevOps, and cloud native. One email a week.')
      : tHome('newsletterHomeBody', 'Practical lessons, tools worth trying, and the week’s key developments across AI, DevOps, and cloud native. One free email, every week.');
    section.innerHTML = `<div class="newsletter-strip">
      <div class="newsletter-copy">
        <h2 id="${id}-title">${title}</h2>
        <p>${body}</p>
      </div>
      <div class="newsletter-signup">
        <form action="${publicationUrl}/subscribe" method="get" aria-describedby="${id}-note">
          <label class="newsletter-email-label" for="${id}-email">${tHome('newsletterEmailLabel', 'Email address')}</label>
          <div class="newsletter-input-row">
            <input id="${id}-email" name="email" type="email" placeholder="${tHome('newsletterEmailPlaceholder', 'Your email address')}" autocomplete="email" required>
            <button type="submit">${tHome('newsletterSubscribe', 'Subscribe free')} <span aria-hidden="true">↗</span></button>
          </div>
        </form>
        <p id="${id}-note" class="newsletter-note">${tHome('newsletterNote', 'Free. Unsubscribe anytime. Finish signup on Substack.')}</p>
      </div>
    </div>`;
    return section;
  }

  function newsletterLink(label) {
    const link = document.createElement('a');
    link.href = publicationUrl + '/subscribe';
    link.textContent = label || tHome('navNewsletter', 'Newsletter');
    link.dataset.newsletterOpen = '';
    if (typeof HTMLDialogElement !== 'undefined' && HTMLDialogElement.prototype.showModal) {
      link.setAttribute('aria-haspopup', 'dialog');
    }
    return link;
  }

  function addNavigation() {
    const nav = document.querySelector('.header-nav');
    if (nav && !nav.querySelector('[data-newsletter-open]')) {
      nav.insertBefore(newsletterLink(), nav.querySelector('.header-mobile-tools'));
    }
    const footer = document.querySelector('.footer-links');
    if (footer && !footer.querySelector('[data-newsletter-open]')) footer.prepend(newsletterLink());
  }

  function createDialog() {
    const element = document.createElement('dialog');
    element.className = 'newsletter-modal';
    element.setAttribute('aria-labelledby', 'newsletter-modal-title');
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'newsletter-modal-close';
    close.textContent = tHome('newsletterClose', 'Close ×');
    close.setAttribute('aria-label', tHome('newsletterCloseAria', 'Close newsletter signup'));
    close.autofocus = true;
    close.addEventListener('click', function () { element.close(); });
    element.append(close, signupSection('dialog'));
    element.addEventListener('click', function (event) {
      if (event.target === element) element.close();
    });
    element.addEventListener('close', function () {
      const focusTarget = trigger?.getClientRects().length ? trigger : document.querySelector('.header-menu-toggle');
      focusTarget?.focus({ preventScroll: true });
    });
    document.body.append(element);
    return element;
  }

  function enableDialog() {
    document.addEventListener('click', function (event) {
      const link = event.target.closest('a[data-newsletter-open]');
      if (!link || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      if (typeof HTMLDialogElement === 'undefined' || !HTMLDialogElement.prototype.showModal) return;
      event.preventDefault();
      trigger = link;
      if (!dialog) dialog = createDialog();
      dialog.showModal();
    });
  }

  function revealAnchor() {
    if (location.hash !== '#newsletter' && location.hash !== '#newsletter-cta') return;
    document.fonts.ready.then(function () {
      requestAnimationFrame(function () {
        document.getElementById(location.hash.slice(1))?.scrollIntoView({ block: 'start', behavior: 'instant' });
      });
    });
  }

  function addPagePlacement() {
    let page = location.pathname.split('/').pop().replace(/\.html$/, '');
    const aliases = { path: 'prereqs', roadmap: 'prereqs', docs: 'developer' };
    page = aliases[page] || page;
    if (page === 'about') {
      document.querySelector('.about')?.append(signupSection('about'));
      return;
    }
    const placements = {
      'learning-paths': ['#overview', 'after', 'Keep building your AI skills, every week.', 'learning-paths-container'],
      catalog: ['.catalog-header', 'append', 'Get practical lessons and the week’s AI updates.'],
      certifications: ['.cert-hero', 'after', 'Keep learning as AI tools and practices evolve.', 'cert-container'],
      certification: ['#trackProgramNotice', 'before', 'Keep building beyond your study plan.', 'cert-container'],
      glossary: ['.glossary-explorer', 'after', 'Turn AI concepts into practical skills.'],
      prereqs: ['.roadmap-workspace', 'after', 'Keep your learning connected to what’s next.', 'roadmap-container'],
      developer: ['.trust-page', 'append', 'Follow the tools and ideas shaping AI engineering.']
    };
    const placement = placements[page];
    if (!placement) return;
    const [selector, position, message, container = ''] = placement;
    const target = document.querySelector(selector);
    if (!target) return;
    const callout = document.createElement('aside');
    callout.id = 'newsletter-cta';
    callout.className = 'newsletter-callout ' + container;
    callout.setAttribute('aria-label', tHome('newsletterAria', 'AI Engineering Newsletter'));
    callout.dataset.newsletterMessage = message;
    const row = document.createElement('div');
    row.className = 'newsletter-callout-row';
    const copy = document.createElement('p');
    copy.textContent = message;
    row.append(copy, newsletterLink(tHome('newsletterCta', 'Get the free newsletter →')));
    callout.append(row);
    target[position](callout);
  }

  function addLessonPlacement() {
    const content = document.getElementById('lessonContent');
    if (!content) return;
    function insert() {
      const next = content.querySelector('.lesson-nav-bottom');
      if (!next || content.querySelector('#newsletter')) return false;
      next.before(signupSection('lesson'));
      revealAnchor();
      return true;
    }
    if (insert()) return;
    const observer = new MutationObserver(function () {
      if (insert()) observer.disconnect();
    });
    observer.observe(content, { childList: true, subtree: true });
  }

  function applyLanguage() {
    document.querySelectorAll('a[data-newsletter-open]').forEach(function (link) {
      if (link.closest('.newsletter-callout')) {
        link.textContent = tHome('newsletterCta', 'Get the free newsletter →');
      } else {
        link.textContent = tHome('navNewsletter', 'Newsletter');
      }
    });

    document.querySelectorAll('.newsletter-section').forEach(function (section) {
      const variant = section.dataset.newsletterVariant || (section.classList.contains('newsletter-lesson') ? 'lesson' : 'home');
      const lesson = variant === 'lesson';
      const title = section.querySelector('h2');
      const body = section.querySelector('.newsletter-copy p');
      const label = section.querySelector('.newsletter-email-label');
      const input = section.querySelector('input[type="email"]');
      const button = section.querySelector('button[type="submit"]');
      const note = section.querySelector('.newsletter-note');
      if (title) {
        title.textContent = lesson
          ? tHome('newsletterLessonTitle', 'AI Engineering Newsletter')
          : tHome('newsletterHomeTitle', 'Learn AI from scratch. Stay ahead of what’s next.');
      }
      if (body) {
        body.textContent = lesson
          ? tHome('newsletterLessonBody', 'Practical lessons and updates on AI, DevOps, and cloud native. One email a week.')
          : tHome('newsletterHomeBody', 'Practical lessons, tools worth trying, and the week’s key developments across AI, DevOps, and cloud native. One free email, every week.');
      }
      if (label) label.textContent = tHome('newsletterEmailLabel', 'Email address');
      if (input) input.placeholder = tHome('newsletterEmailPlaceholder', 'Your email address');
      if (button) {
        button.innerHTML = tHome('newsletterSubscribe', 'Subscribe free') + ' <span aria-hidden="true">↗</span>';
      }
      if (note) {
        note.textContent = tHome('newsletterNote', 'Free. Unsubscribe anytime. Finish signup on Substack.');
      }
    });

    const callout = document.getElementById('newsletter-cta');
    if (callout) {
      callout.setAttribute('aria-label', tHome('newsletterAria', 'AI Engineering Newsletter'));
      const cta = callout.querySelector('a[data-newsletter-open]');
      if (cta) cta.textContent = tHome('newsletterCta', 'Get the free newsletter →');
    }

    if (dialog) {
      const close = dialog.querySelector('.newsletter-modal-close');
      if (close) {
        close.textContent = tHome('newsletterClose', 'Close ×');
        close.setAttribute('aria-label', tHome('newsletterCloseAria', 'Close newsletter signup'));
      }
    }

    const colophon = document.querySelector('.colophon p');
    if (colophon && /No paywall, no (signup|account required)\./.test(colophon.textContent)) {
      const koNote = tHome('colophonNoPaywall', 'No paywall, no account required.');
      colophon.textContent = colophon.textContent
        .replace(/No paywall, no signup\./, koNote)
        .replace(/No paywall, no account required\./, koNote);
    }
  }

  function initialize() {
    if (!document.querySelector('.site-header') || document.querySelector('[data-newsletter-open]')) return;
    addNavigation();
    enableDialog();
    const readers = document.querySelector('.learners-strip');
    if (readers) {
      readers.before(signupSection('home'));
      const colophon = document.querySelector('.colophon p');
      if (colophon) {
        colophon.textContent = colophon.textContent.replace(
          'No paywall, no signup.',
          tHome('colophonNoPaywall', 'No paywall, no account required.')
        );
      }
    } else {
      addPagePlacement();
      addLessonPlacement();
    }
    applyLanguage();
    revealAnchor();
  }

  document.addEventListener('aifs:lang', applyLanguage);
  window.AIFS_applyNewsletterLang = applyLanguage;

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize);
  else initialize();
})();
